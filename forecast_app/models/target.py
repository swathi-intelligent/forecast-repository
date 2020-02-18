import datetime
import itertools

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import BooleanField, IntegerField

from forecast_app.models import Project, PointPrediction, BinDistribution, SampleDistribution, NamedDistribution
from utils.utilities import basic_str, YYYY_MM_DD_DATE_FORMAT


#
# ---- Target ----
#

class Target(models.Model):
    """
    Represents one of a Project's targets. See https://github.com/reichlab/docs.zoltardata/ for details about
    target_type and related information.
    """
    # database-level data_types - leveraging Python's built-in types
    BOOLEAN_DATA_TYPE = bool
    DATE_DATA_TYPE = datetime.date
    FLOAT_DATA_TYPE = float
    INTEGER_DATA_TYPE = int
    TEXT_DATA_TYPE = str

    # date unit choices
    DATE_UNITS = ['month', 'week', 'biweek', 'day']

    project = models.ForeignKey(Project, related_name='targets', on_delete=models.CASCADE)

    # target_type choices
    CONTINUOUS_TARGET_TYPE = 0
    DISCRETE_TARGET_TYPE = 1
    NOMINAL_TARGET_TYPE = 2
    BINARY_TARGET_TYPE = 3
    DATE_TARGET_TYPE = 4
    TARGET_TYPE_CHOICES = (
        (CONTINUOUS_TARGET_TYPE, 'continuous'),
        (DISCRETE_TARGET_TYPE, 'discrete'),
        (NOMINAL_TARGET_TYPE, 'nominal'),
        (BINARY_TARGET_TYPE, 'binary'),
        (DATE_TARGET_TYPE, 'date'),
    )
    # required fields for all types
    type = models.IntegerField(choices=TARGET_TYPE_CHOICES,
                               help_text="The Target's type. The choices are 'continuous', 'discrete', 'nominal', "
                                         "'binary', and 'date'.")
    name = models.TextField(help_text="A brief name for the target.")
    description = models.TextField(help_text="A verbose description of what the target is.")
    is_step_ahead = BooleanField(help_text="True if the target is one of a sequence of targets that predict values at "
                                           "different points in the future.")
    step_ahead_increment = IntegerField(help_text="An integer, indicating the forecast horizon represented by this "
                                                  "target. It is required if `is_step_ahead` is True.",
                                        null=True, default=None)

    # type-specific fields
    unit = models.TextField(help_text="This target's units, e.g., 'percentage', 'week', 'cases', etc.", null=True)


    # 'list' type-specific fields: see TargetLwr.lwrs, TargetCat.cats, and TargetDate.range


    def __repr__(self):
        return str((self.pk, self.name, Target.str_for_target_type(self.type),
                    self.is_step_ahead, self.step_ahead_increment))


    def __str__(self):  # todo
        return basic_str(self)


    def type_as_str(self):
        return Target.str_for_target_type(self.type)


    @classmethod
    def str_for_target_type(cls, the_type_int):
        for type_int, type_name in cls.TARGET_TYPE_CHOICES:
            if type_int == the_type_int:
                return type_name

        return '!?'


    def save(self, *args, **kwargs):
        """
        Validates is_step_ahead and step_ahead_increment, and is_date and is_step_ahead.
        """
        from utils.project import _target_dict_for_target, _validate_target_dict  # avoid circular imports


        # validate by serializing to a dict so we can use _validate_target_dict(). note that Targets created without
        # a name, description
        type_name_to_type_int = {type_name: type_int for type_int, type_name in Target.TARGET_TYPE_CHOICES}
        target_dict = _target_dict_for_target(self)
        _validate_target_dict(target_dict, type_name_to_type_int)  # raises RuntimeError if invalid

        # done
        super().save(*args, **kwargs)


    def data_type(self):
        return Target.data_type_for_target_type(self.type)


    @classmethod
    def data_type_for_target_type(cls, target_type):
        """
        :param target_type: one of my *_TARGET_TYPE values
        :return: the database data_type for target_type
        """
        return {
            Target.CONTINUOUS_TARGET_TYPE: Target.FLOAT_DATA_TYPE,
            Target.DISCRETE_TARGET_TYPE: Target.INTEGER_DATA_TYPE,
            Target.NOMINAL_TARGET_TYPE: Target.TEXT_DATA_TYPE,
            Target.BINARY_TARGET_TYPE: Target.BOOLEAN_DATA_TYPE,
            Target.DATE_TARGET_TYPE: Target.DATE_DATA_TYPE,
        }[target_type]


    @classmethod
    def is_value_compatible_with_target_type(cls, target_type, value):
        """
        Returns True if value's type is compatible with target_type. NB: for date target types, will try to parse first,
        but will not raise an error if that fails.

        :param target_type: one of my *_TARGET_TYPE values
        :param value: an int, float, str, or boolean
        """
        value_type = type(value)
        if (target_type == Target.CONTINUOUS_TARGET_TYPE) and \
                ((value_type == Target.data_type_for_target_type(Target.CONTINUOUS_TARGET_TYPE)) or
                 (value_type == Target.data_type_for_target_type(Target.DISCRETE_TARGET_TYPE))):
            return True
        elif (target_type == Target.DISCRETE_TARGET_TYPE) and \
                (value_type == Target.data_type_for_target_type(Target.DISCRETE_TARGET_TYPE)):
            return True
        elif (target_type == Target.NOMINAL_TARGET_TYPE) and \
                (value_type == Target.data_type_for_target_type(Target.NOMINAL_TARGET_TYPE)):
            return True
        elif (target_type == Target.BINARY_TARGET_TYPE) and \
                (value_type == Target.data_type_for_target_type(Target.BINARY_TARGET_TYPE)):
            return True
        elif (target_type == Target.DATE_TARGET_TYPE) and (value_type == str):
            try:
                datetime.datetime.strptime(value, YYYY_MM_DD_DATE_FORMAT).date()
                return True
            except ValueError:
                return False

        return False


    def set_cats(self, cats, extra_lwr=None):
        """
        Creates TargetCat and optional TargetLwr entries for each cat in cats, first deleting all current ones.

        :param cats: a list of categories. they are either all ints, floats, or strs depending on my data_type. strs
            will be converted to datetime.date objects for date targets.
        :param extra_lwr: an optional final upper lwr to use when creating TargetLwrs. used when a Target has both cats
            and range
        """
        # validate uniform data type
        data_type = self.data_type()
        types_set = set(map(type, cats))
        if len(types_set) != 1:
            raise ValidationError(f"there was more than one data type in cats={cats}: {types_set}")

        # before validating data type compatibility, try to replace date strings with actual date objects
        try:
            if data_type == Target.DATE_DATA_TYPE:
                cats = [datetime.datetime.strptime(cat_str, YYYY_MM_DD_DATE_FORMAT).date() for cat_str in cats]
        except ValueError as exc:
            raise ValidationError(f"one or more cats were not in YYYY-MM-DD format. cats={cats}. exc={exc}")

        # validate data type compatibility
        types_set = set(map(type, cats))
        cats_type = next(iter(types_set))  # vs. pop()
        if data_type != cats_type:
            raise ValidationError(f"cats data type did not match target data type. cats={cats}. cats_type={cats_type}, "
                                  f"data_type={data_type}")

        # delete and save the new TargetCats
        TargetCat.objects.filter(target=self).delete()
        for cat in cats:
            TargetCat.objects.create(target=self,
                                     cat_i=cat if (data_type == Target.INTEGER_DATA_TYPE) else None,
                                     cat_f=cat if (data_type == Target.FLOAT_DATA_TYPE) else None,
                                     cat_t=cat if (data_type == Target.TEXT_DATA_TYPE) else None,
                                     cat_d=cat if (data_type == Target.DATE_DATA_TYPE) else None,
                                     cat_b=cat if (data_type == Target.BOOLEAN_DATA_TYPE) else None)

        # ditto for TargetLwrs for continuous and discrete cases (required for scoring), calculating `upper` via zip().
        # NB: we use infinity for the last bin's upper!
        if self.type in [Target.CONTINUOUS_TARGET_TYPE, Target.DISCRETE_TARGET_TYPE]:
            cats = sorted(cats)
            if extra_lwr:
                cats.append(extra_lwr)
            for lwr, upper in itertools.zip_longest(cats, cats[1:], fillvalue=float('inf')):
                TargetLwr.objects.create(target=self, lwr=lwr, upper=upper)


    def set_range(self, lower, upper):
        """
        Creates two TargetRange entries for lower and upper, first deleting all current ones.

        :param lower: an int or float, depending on my data_type
        :param upper: ""
        """
        # validate target type
        valid_target_types = [Target.CONTINUOUS_TARGET_TYPE, Target.DISCRETE_TARGET_TYPE]
        if self.type not in valid_target_types:
            raise ValidationError(f"invalid target type '{self.type}'. range must be one of: {valid_target_types}")

        # validate lower, upper
        data_type = self.data_type()
        if type(lower) != type(upper):
            raise ValidationError(f"lower and upper were of different data types: {type(lower)} != {type(upper)}")
        elif data_type != type(lower):
            raise ValidationError(f"lower and upper data type did not match target data type. "
                                  f"lower/upper type={type(lower)}, data_type={data_type}")

        # delete and save the new TargetRanges
        TargetRange.objects.filter(target=self).delete()
        TargetRange.objects.create(target=self,
                                   value_i=lower if (data_type == Target.INTEGER_DATA_TYPE) else None,
                                   value_f=lower if (data_type == Target.FLOAT_DATA_TYPE) else None)
        TargetRange.objects.create(target=self,
                                   value_i=upper if (data_type == Target.INTEGER_DATA_TYPE) else None,
                                   value_f=upper if (data_type == Target.FLOAT_DATA_TYPE) else None)


    def range_tuple(self):
        """
        :return: either a 2-tuple () if I have a ranges, or None o/w. ordered by min, max
        """
        ranges_qs = self.ranges.all()
        if not ranges_qs.count():
            return

        ranges_list = list(ranges_qs)
        ranges0 = ranges_list[0]
        ranges1 = ranges_list[1]
        ranges0_val = PointPrediction.first_non_none_value(ranges0.value_i, ranges0.value_f, None, None, None)
        ranges1_val = PointPrediction.first_non_none_value(ranges1.value_i, ranges1.value_f, None, None, None)
        return min(ranges0_val, ranges1_val), max(ranges0_val, ranges1_val)


    def cats_values(self):
        """
        A utility function used for validation. Returns a list of my cat values based on my data_type(), similar to what
        PointPrediction.first_non_none_value() might do, except instead of retrieving all cat_* fields we only get the
        field corresponding to my type.
        """
        data_type = self.data_type()
        if data_type == Target.INTEGER_DATA_TYPE:
            values = self.cats.values_list('cat_i', flat=True)
        elif data_type == Target.FLOAT_DATA_TYPE:
            values = self.cats.values_list('cat_f', flat=True)
        elif data_type == Target.TEXT_DATA_TYPE:
            values = self.cats.values_list('cat_t', flat=True)
        elif data_type == Target.DATE_DATA_TYPE:
            values = self.cats.values_list('cat_d', flat=True)
        else:  # data_type == Target.BINARY_TARGET_TYPE
            values = self.cats.values_list('cat_b', flat=True)
        return list(values)


    @classmethod
    def valid_named_families(cls, target_type):
        """
        :param target_type: one of my *_TARGET_TYPE values
        :return: a list of valid NamedDistribution families for target_type
        """
        return {
            Target.CONTINUOUS_TARGET_TYPE: [NamedDistribution.NORM_DIST, NamedDistribution.LNORM_DIST,
                                            NamedDistribution.GAMMA_DIST, NamedDistribution.BETA_DIST],
            Target.DISCRETE_TARGET_TYPE: [NamedDistribution.POIS_DIST, NamedDistribution.NBINOM_DIST,
                                          NamedDistribution.NBINOM2_DIST],
            Target.NOMINAL_TARGET_TYPE: [],  # n/a
            Target.BINARY_TARGET_TYPE: [],  # n/a
            Target.DATE_TARGET_TYPE: [],  # n/a
        }[target_type]


    @classmethod
    def valid_prediction_types(cls, target_type):
        """
        :param target_type: one of my *_TARGET_TYPE values
        :return: a list of valid concrete Prediction subclasses for target_type
        """
        return {
            Target.CONTINUOUS_TARGET_TYPE: [PointPrediction, BinDistribution, SampleDistribution, NamedDistribution],
            Target.DISCRETE_TARGET_TYPE: [PointPrediction, BinDistribution, SampleDistribution, NamedDistribution],
            Target.NOMINAL_TARGET_TYPE: [PointPrediction, BinDistribution, SampleDistribution],
            Target.BINARY_TARGET_TYPE: [PointPrediction, SampleDistribution, NamedDistribution],
            Target.DATE_TARGET_TYPE: [PointPrediction, BinDistribution, SampleDistribution],
        }[target_type]


#
# ---- TargetCat ----
#

class TargetCat(models.Model):
    """
    Associates a 'list' of cat values with Targets of type Target.CONTINUOUS_TARGET_TYPE, Target.DISCRETE_TARGET_TYPE,
    Target.NOMINAL_TARGET_TYPE, or Target.DATE_TARGET_TYPE.
    """
    target = models.ForeignKey('Target', blank=True, null=True, related_name='cats', on_delete=models.CASCADE)
    cat_i = models.IntegerField(null=True)  # NULL if any others non-NULL
    cat_f = models.FloatField(null=True)  # ""
    cat_t = models.TextField(null=True)  # ""
    cat_d = models.DateField(null=True)  # ""
    cat_b = models.NullBooleanField(null=True)  # ""


    def __repr__(self):
        return str((self.pk, self.target.pk, self.cat_i, self.cat_f, self.cat_t, self.cat_d, self.cat_b))


    def __str__(self):  # todo
        return basic_str(self)


#
# ---- TargetLwr ----
#

class TargetLwr(models.Model):
    """
    Associates a 'list' of lwr values with Targets of type Target.CONTINUOUS_TARGET_TYPE that have 'cats'

    . These act as a "template"
    against which forecast TargetLwr predictions can be validated against. Note that only lwr is typically passed by the
    user (as `cat`). upper is typically calculated from lwr by the caller.

    Regarding upper: It is currently used only for scoring, when the true bin is queried for. In that case we test
    truth >= lwr AND truth < upper. Therefore it is currently calculated by utils.project.validate_and_create_targets()
    based on lwr. That function has to infer the final bin's upper, and uses float('inf') for that
    """

    target = models.ForeignKey('Target', blank=True, null=True, related_name='lwrs', on_delete=models.CASCADE)
    lwr = models.FloatField(null=True)  # nullable b/c some bins have non-numeric values, e.g., 'NA'
    upper = models.FloatField(null=True)  # "". possibly float('inf')


    def __repr__(self):
        return str((self.pk, self.target.pk, self.lwr, self.upper))


    def __str__(self):  # todo
        return basic_str(self)


#
# ---- TargetRange ----
#

class TargetRange(models.Model):
    """
    Associates a 'list' of range values with Targets of type Target.CONTINUOUS_TARGET_TYPE or Target.DISCRETE_TARGET_TYPE. Note that
    unlike other 'list' Models relating to Target, this one should have exactly two rows per target, where the first
    one's value is the lower range number, and the second row's value is the upper range number.
    """
    target = models.ForeignKey('Target', blank=True, null=True, related_name='ranges', on_delete=models.CASCADE)
    value_i = models.IntegerField(null=True)  # NULL if value_f is non-NULL
    value_f = models.FloatField(null=True)  # "" value_i ""


    def __repr__(self):
        return str((self.pk, self.target.pk, self.value_i, self.value_f))


    def __str__(self):  # todo
        return basic_str(self)
