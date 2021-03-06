import logging
from collections import defaultdict
from itertools import groupby

from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import ManyToManyField, Max
from django.urls import reverse

from utils.utilities import basic_str


logger = logging.getLogger(__name__)

#
# ---- Project class ----
#

POSTGRES_NULL_VALUE = 'NULL'  # used for Postgres-specific loading of rows from csv data files

TRUTH_CSV_HEADER = ['timezero', 'unit', 'target', 'value']


class Project(models.Model):
    """
    The make_cdc_flu_contests_project_app class representing a forecast challenge, including metadata, core data,
    targets, and model entries.
    """

    # w/out related_name we get: forecast_app.Project.model_owners:
    #   (fields.E304) Reverse accessor for 'Project.model_owners' clashes with reverse accessor for 'Project.owner'.
    owner = models.ForeignKey(User,
                              related_name='project_owner',
                              on_delete=models.SET_NULL,
                              blank=True, null=True,
                              help_text="The project's owner.")

    is_public = models.BooleanField(default=True,
                                    help_text="Controls project visibility. False means the project is private and "
                                              "can only be accessed by the project's owner or any of its model_owners. "
                                              "True means it is publicly accessible.")

    model_owners = ManyToManyField(User, blank=True,
                                   help_text="Users who are allowed to create, edit, and delete ForecastModels "
                                             "in this project. Or: non-editing users who simply need access "
                                             "to a private project. Use control/command click to add/remove from "
                                             "the list. ")

    name = models.TextField()

    WEEK_TIME_INTERVAL_TYPE = 'w'
    BIWEEK_TIME_INTERVAL_TYPE = 'b'
    MONTH_TIME_INTERVAL_TYPE = 'm'
    TIME_INTERVAL_TYPE_CHOICES = ((WEEK_TIME_INTERVAL_TYPE, 'Week'),
                                  (BIWEEK_TIME_INTERVAL_TYPE, 'Biweek'),
                                  (MONTH_TIME_INTERVAL_TYPE, 'Month'))
    time_interval_type = models.CharField(max_length=1,
                                          choices=TIME_INTERVAL_TYPE_CHOICES, default=WEEK_TIME_INTERVAL_TYPE,
                                          help_text="Used when visualizing the x axis label.")
    visualization_y_label = models.TextField(help_text="Used when visualizing the Y axis label.")

    truth_csv_filename = models.TextField(help_text="Name of the truth csv file that was uploaded.")
    truth_updated_at = models.DateTimeField(blank=True, null=True, help_text="The last time the truth was updated.")

    description = models.TextField(help_text="A few paragraphs describing the project. Please see documentation for"
                                             "what should be included here - 'real-time-ness', time_zeros, etc.")

    home_url = models.URLField(help_text="The project's home site.")
    logo_url = models.URLField(blank=True, null=True, help_text="The project's optional logo image.")
    core_data = models.URLField(
        help_text="Directory or Zip file containing data files (e.g., CSV files) made made available to everyone in "
                  "the challenge, including supplemental data like Google queries or weather.")


    def __repr__(self):
        return str((self.pk, self.name))


    def __str__(self):  # todo
        return basic_str(self)


    def save(self, *args, **kwargs):
        """
        Validates my TimeZero.timezero_dates for uniqueness.
        """
        found_timezero_dates = []
        for timezero in self.timezeros.all():
            if timezero.timezero_date not in found_timezero_dates:
                found_timezero_dates.append(timezero.timezero_date)
            else:
                raise ValidationError("found duplicate TimeZero.timezero_date: {}".format(timezero.timezero_date))

        # done
        super().save(*args, **kwargs)


    def time_interval_type_as_str(self):
        """
        :return: my time_interval_type as a human-friendly string from TIME_INTERVAL_TYPE_CHOICES
        """
        for db_value, human_readable_value in Project.TIME_INTERVAL_TYPE_CHOICES:
            if db_value == self.time_interval_type:
                return human_readable_value


    def get_absolute_url(self):
        return reverse('project-detail', args=[str(self.pk)])


    def get_class(self):
        """
        :return: view utility that simply returns a my class as a string. used by delete_modal_snippet.html
        """
        return self.__class__.__name__


    def html_id(self):
        """
        :return: view utility that returns a unique HTML id for this object. used by delete_modal_snippet.html
        """
        return self.__class__.__name__ + '_' + str(self.pk)


    #
    # season-related utilities
    #

    def seasons(self):
        """
        :return: list of season names for this project based on my timezeros
        """
        return list(self.timezeros
                    .filter(is_season_start=True)
                    .order_by('timezero_date')
                    .values_list('season_name', flat=True))


    def timezeros_in_season(self, season_name):
        """
        Utility that returns a sorted list of TimeZeros for season_name.

        :param season_name: a valid season name (see seasons()) or None, which is used to access TimeZeros that have
            no season. For the latter, there are two cases:
            1) there are no seasons at all
            2) there are seasons, but the first starts after the first TimeZero, i.e., my TimeZeros start with some
               non-season ones that are followed by some seasons
        :return: two cases based on whether season_name is None. 1) If not None: returns a list of TimeZeros that are
            within season_name, i.e., those that start with the TimeZero named season_name and go TO the next season,
            or to the end if season_name is the last season. 2) If None: returns based on the two cases listed above
            for season_name: 1) no seasons at all: return all TimeZeros. 2) starts with some non-seasons: return those
            up TO the first season.
        """
        # start with all TimeZeros - case #1 (no seasons at all), and filter as needed
        season_timezeros_qs = self.timezeros.all()
        if season_name:
            season_tz = season_timezeros_qs.filter(season_name=season_name).first()
            if not season_tz:
                raise RuntimeError("invalid season_name. season_name={}, seasons={}"
                                   .format(season_name, self.seasons()))

            season_timezeros_qs = season_timezeros_qs.filter(timezero_date__gte=season_tz.timezero_date)
            next_season_tz = season_timezeros_qs \
                .filter(is_season_start=True,
                        timezero_date__gt=season_tz.timezero_date) \
                .first()
            if next_season_tz:
                season_timezeros_qs = season_timezeros_qs.filter(timezero_date__lt=next_season_tz.timezero_date)
        else:  # no season_name
            first_season_tz = season_timezeros_qs.filter(is_season_start=True).first()
            if first_season_tz:  # case #2 (seasons after initial TZs)
                season_timezeros_qs = season_timezeros_qs.filter(timezero_date__lt=first_season_tz.timezero_date)
        return list(season_timezeros_qs.order_by('timezero_date'))


    def start_end_dates_for_season(self, season_name):
        """
        :param season_name: same as timezeros_in_season() - can be None
        :return: 2-tuple: (start_date, end_date) for season_name. this is a closed interval - both are included.
            Note that start_date == end_date if there is only one TimeZero. returns None if no TimeZeros found
        """
        timezeros = self.timezeros_in_season(season_name)
        if len(timezeros) == 0:
            return None

        return timezeros[0].timezero_date, timezeros[-1].timezero_date


    def season_name_containing_timezero(self, timezero, timezeros=None):
        """
        :return: season_name of the season that contains timezero, or None if it's not in a season. timezeros, if
            passed, allows optimizing by callers who compute timezeros only once.
        """
        timezeros = timezeros or self.timezeros.all()
        if timezero not in timezeros:
            raise RuntimeError("TimeZero not found in timezeros: timezero={}, timezeros={}".format(timezero, timezeros))

        # order my timezeros by date and then iterate from earliest to latest, keeping track of the current season and
        # returning the first match. must handle two cases: the earliest timezero defines a season, or not
        containing_season_name = None  # return value. updated in loop
        for project_timezero in timezeros.order_by('timezero_date'):
            if project_timezero.is_season_start:
                containing_season_name = project_timezero.season_name
            if project_timezero == timezero:
                return containing_season_name


    def timezero_to_season_name(self):
        """
        :return: a dict mapping each of my timezeros -> containing season name
        """
        _timezero_to_season_name = {}
        containing_season_name = None
        for timezero in self.timezeros.order_by('timezero_date'):
            if timezero.is_season_start:
                containing_season_name = timezero.season_name
            _timezero_to_season_name[timezero] = containing_season_name
        return _timezero_to_season_name


    #
    # time-related utilities
    #

    def forecasts_for_timezero(self, timezero):
        """
        :param timezero: a TimeZero
        :return: a list of Forecasts for timezero for each of my models
        """
        return [forecast_model.forecast_for_time_zero(timezero) for forecast_model in self.models.all()]


    def time_zero_for_timezero_date(self, timezero_date):
        """
        :return: the first TimeZero in me that has a timezero_date matching timezero_date
        """
        return self.timezeros.filter(timezero_date=timezero_date).first()


    def time_interval_type_to_foresight(self):
        """
        :return: my time_interval_type formatted for D3-Foresight's pointType
        """
        return dict(Project.TIME_INTERVAL_TYPE_CHOICES)[self.time_interval_type].lower()


    def last_update(self):
        """
        Returns the datetime.datetime of the last time this project was "updated". currently only uses
        Project.truth_updated_at, and Forecast.created_at for all models in me.
        """
        from .forecast import Forecast  # avoid circular imports


        latest_forecast = Forecast.objects.filter(forecast_model__project=self).order_by('-created_at').first()
        update_dates = [self.truth_updated_at, latest_forecast.created_at if latest_forecast else None]
        # per https://stackoverflow.com/questions/19868767/how-do-i-sort-a-list-with-nones-last
        return sorted(update_dates, key=lambda _: (_ is not None, _))[-1]


    #
    # count-related functions
    #

    def get_summary_counts(self):
        """
        :return: a 3-tuple summarizing total counts in me: (num_models, num_forecasts, num_rows). The latter is
            estimated.
        """
        from .forecast import Forecast  # avoid circular imports


        return self.models.filter(project=self).count(), \
               Forecast.objects.filter(forecast_model__project=self).count(), \
               self.get_num_forecast_rows_all_models_estimated()


    def get_num_forecast_rows_all_models(self):
        """
        :return: the total number of data rows across all my models' forecasts, for all types of Predictions. can be
        expensive for large databases
        """
        from forecast_app.models import Prediction  # avoid circular imports


        return sum(concrete_prediction_class.objects.filter(forecast__forecast_model__project=self).count()
                   for concrete_prediction_class in Prediction.concrete_subclasses())


    def get_num_forecast_rows_all_models_estimated(self):
        """
        :return: like get_num_forecast_rows_all_models(), but returns an estimate that is much faster to calculate. the estimate
            is based on getting the number of rows for an arbitrary Forecast and then multiplying by the number of
            forecasts times the number of models in me. it will be exact for projects whose models all have the same
            number of rows
        """
        first_model = self.models.first()
        first_forecast = first_model.forecasts.first() if first_model else None
        first_forecast_num_rows = first_forecast.get_num_rows() if first_forecast else None
        return (self.models.count() * first_model.forecasts.count() * first_forecast_num_rows) \
            if first_forecast_num_rows else 0


    def unit_to_max_val(self, season_name, targets):
        """
        :return: a dict mapping each unit_name to the maximum point value across all my forecasts for season_name
            and targets
        """
        from forecast_app.models import PointPrediction  # avoid circular imports


        # NB: we retrieve and max() only the two numeric value fields (value_i and value_f), excluding value_t (which
        # has no meaningful max() semantics). a concern is that some targets in the results might have a
        # point_value_type of POINT_INTEGER while others are POINT_FLOAT, but this shouldn't matter to our callers, who
        # are simply trying to get the maximum across /all/ targets. I think.
        season_start_date, season_end_date = self.start_end_dates_for_season(season_name)
        loc_max_val_qs = PointPrediction.objects \
            .filter(forecast__forecast_model__project=self,
                    target__in=targets,
                    forecast__time_zero__timezero_date__gte=season_start_date,
                    forecast__time_zero__timezero_date__lte=season_end_date) \
            .values('unit__name') \
            .annotate(Max('value_i'), Max('value_f'))  # values() -> annotate() is a GROUP BY
        # [{'unit__name': 'HHS Region 1', 'value_i__max': None, 'value_f__max': 2.06145600601835}, ...]

        # per https://stackoverflow.com/questions/12229902/sum-a-list-which-contains-none-using-python :
        return {loc_max_val_dict['unit__name']: max(filter(None, [loc_max_val_dict['value_i__max'],
                                                                  loc_max_val_dict['value_f__max']]))
                for loc_max_val_dict in loc_max_val_qs}


    #
    # visualization-related functions
    #

    def step_ahead_targets(self):
        return self.targets.filter(is_step_ahead=True) \
            .order_by('name')


    def numeric_targets(self):
        """
        :return: a list of Targets whose values are numeric - either int or float. used by scoring
        """
        from forecast_app.models import Target  # avoid circular imports


        return self.targets.filter(type__in=[Target.CONTINUOUS_TARGET_TYPE, Target.DISCRETE_TARGET_TYPE]) \
            .order_by('name')


    #
    # truth data-related functions
    #

    def is_truth_data_loaded(self):
        """
        :return: True if I have truth data loaded via load_truth_data(). Actually, returns the count, which acts as a
            boolean.
        """
        return self.truth_data_qs().exists()


    def get_truth_data_preview(self):
        """
        :return: view helper function that returns a preview of my truth data in the form of a table that's represented
            as a nested list of rows. each row: [timezero_date, unit_name, target_name, truth_value]
        """
        from forecast_app.models import PointPrediction  # avoid circular imports


        rows = self.truth_data_qs().values_list('time_zero__timezero_date', 'unit__name', 'target__name',
                                                'value_i', 'value_f', 'value_t', 'value_d', 'value_b')[:10]
        return [[timezero_date, unit_name, target_name,
                 PointPrediction.first_non_none_value(value_i, value_f, value_t, value_d, value_b)]
                for timezero_date, unit_name, target_name, value_i, value_f, value_t, value_d, value_b in rows]


    def get_num_truth_rows(self):
        return self.truth_data_qs().count()


    def get_truth_data_rows(self):
        """
        Returns all of my data as a a list of rows, excluding any PKs and FKs columns, and ordered by PK.
        """
        return list(self.truth_data_qs()
                    .order_by('id')
                    .values_list('time_zero__timezero_date', 'unit__name', 'target__name',
                                 'value_i', 'value_f', 'value_t', 'value_d', 'value_b'))


    def truth_data_qs(self):
        """
        :return: A QuerySet of my TruthData.
        """
        from forecast_app.models import TruthData  # avoid circular imports


        return TruthData.objects.filter(time_zero__project=self)


    def delete_truth_data(self):
        self.truth_data_qs().delete()
        self.truth_csv_filename = ''
        self.truth_updated_at = None
        self.save()
        self._update_model_score_changes()


    def reference_target_for_actual_values(self):
        """
        Returns the target in me that should act as the one to use when computing an 'actual' step-ahead value from
        loaded truth data. We try to use the one that is the fewest step ahead steps available, starting with zero and
        going up from there. Returns None if no appropriate targets were found, say if there are no targets, or only
        negative ones.

        _About calculating 'actual' step-head values from truth data_: Loaded truth data contains actual values by way
        of the project's 'step ahead' targets. Some projects provide a zero step ahead target (whose
        step_ahead_increment is 0), which is what we need to get the an actual value for a particular
        [unit][timezero_date] combination: Just index in to the desired timezero_date. However, other projects
        provide only non-zero targets, e.g., '1 wk ahead' (whose step_ahead_increment is 1). In these cases we need a
        'reference' target to use, which we then apply to move that many steps ahead in the project's TimeZeros (sorted
        by date) to get the actual (0 step ahead) value for that timezero_date. For example, if we wan the actual value
        for this truth data:

            timezero   unit       target       value
            20170723   HHS Region 1   1 wk ahead   0.303222
            20170730   HHS Region 1   1 wk ahead   0.286054

        And if we are using '1 wk ahead' as our reference target, then to get the actual step-ahead value for the
        [unit][timezero_date] combination of ['20170730']['HHS Region 1'] we need to work backwards 1
        step_ahead_increment to ['20170723']['HHS Region 1'] and use the '1 wk ahead' target's value, i.e., 0.303222. In
        our example above, there is actual step-ahead value for 20170723.

        Generally, the definition is:
            actual[unit][timezero_date] = truth[unit][ref_target][timezero_date - ref_target_incr]
        """
        from forecast_app.models import Target  # avoid circular imports


        return Target.objects.filter(project=self, is_step_ahead=True, step_ahead_increment__gte=0) \
            .order_by('step_ahead_increment') \
            .first()


    def unit_target_name_tz_date_to_truth(self, season_name=None):
        """
        Returns my truth values as a dict that's organized for easy access, as in:
        unit_target_name_tz_date_to_truth[unit_name][target_name][timezero_date]. Only includes data from
        season_name, which is None if I have no seasons.
        """
        from forecast_app.models import PointPrediction  # avoid circular imports


        logger.debug(f"unit_target_name_tz_date_to_truth(): entered. project={self}, season_name={season_name}")
        loc_target_tz_date_to_truth = {}
        # NB: ordering by target__id is arbitrary. it could be target__name, but it doesn't matter as long it's grouped
        # at all for the second groupby() call below
        truth_data_qs = self.truth_data_qs() \
            .order_by('unit__name', 'target__name') \
            .values_list('unit__id', 'target__id', 'time_zero__timezero_date',
                         'value_i', 'value_f', 'value_t', 'value_d', 'value_b')
        if season_name:
            season_start_date, season_end_date = self.start_end_dates_for_season(season_name)
            truth_data_qs = truth_data_qs.filter(time_zero__timezero_date__gte=season_start_date,
                                                 time_zero__timezero_date__lte=season_end_date)

        unit_pks_to_names = {unit.id: unit.name for unit in self.units.all()}
        target_pks_to_names = {target.id: target.name for target in self.targets.all()}
        for unit_id, loc_target_tz_grouper in groupby(truth_data_qs, key=lambda _: _[0]):
            if unit_id not in unit_pks_to_names:
                continue

            target_tz_date_to_truth = {}
            loc_target_tz_date_to_truth[unit_pks_to_names[unit_id]] = target_tz_date_to_truth
            for target_id, target_tz_grouper in groupby(loc_target_tz_grouper, key=lambda _: _[1]):
                if target_id not in target_pks_to_names:
                    continue

                tz_date_to_truth = defaultdict(list)
                target_tz_date_to_truth[target_pks_to_names[target_id]] = tz_date_to_truth
                for _, _, tz_date, value_i, value_f, value_t, value_d, value_b in target_tz_grouper:
                    value = PointPrediction.first_non_none_value(value_i, value_f, value_t, value_d, value_b)
                    tz_date_to_truth[tz_date].append(value)
        logger.debug(f"unit_target_name_tz_date_to_truth(): done ({len(loc_target_tz_date_to_truth)}). "
                     f"project={self}, season_name={season_name}")
        return loc_target_tz_date_to_truth


    #
    # actual data-related functions
    #

    def unit_timezero_date_to_actual_vals(self, season_name):
        """
        Returns 'actual' step-ahead values from loaded truth data as a dict that's organized for easy access, as in:
        unit_timezero_date_to_actual_vals[unit][timezero_date] . Returns {} if no
        reference_target_for_actual_values().

        :param season_name: optional season. None means return all data
        """


        def is_tz_date_in_season(timezero_date):
            return (timezero_date >= season_start_date) and (timezero_date <= season_end_date)


        ref_target = self.reference_target_for_actual_values()
        if not ref_target:
            return {}

        if season_name:
            season_start_date, season_end_date = self.start_end_dates_for_season(season_name)

        # build tz_date_to_next_tz_date by zipping ordered TimeZeros, staggered by the ref_target's step_ahead_increment
        tz_dates = TimeZero.objects.filter(project=self) \
            .order_by('timezero_date') \
            .values_list('timezero_date', flat=True)
        tz_date_to_next_tz_date = dict(zip(tz_dates, tz_dates[ref_target.step_ahead_increment:]))

        # get loc_target_tz_date_to_truth(). we use all seasons b/c might need TimeZero from a previous season to get
        # this one. recall: [unit][target_name][timezero_date] -> truth
        loc_target_tz_date_to_truth = self.unit_target_name_tz_date_to_truth()  # target__id
        loc_tz_date_to_actual_vals = {}  # [unit][timezero_date] -> actual
        for unit in loc_target_tz_date_to_truth:
            # default to None so that any TimeZeros missing from loc_target_tz_date_to_truth are present:
            unit_dict = {}
            for timezero in tz_dates:
                if not season_name or is_tz_date_in_season(timezero):
                    unit_dict[timezero] = None
            loc_tz_date_to_actual_vals[unit] = unit_dict
            for truth_tz_date in loc_target_tz_date_to_truth[unit][ref_target.name]:
                if truth_tz_date not in tz_date_to_next_tz_date:  # trying to project beyond last truth date
                    continue

                actual_tz_date = tz_date_to_next_tz_date[truth_tz_date]
                truth_value = loc_target_tz_date_to_truth[unit][ref_target.name][truth_tz_date]
                is_actual_in_season = is_tz_date_in_season(actual_tz_date) if season_name else True
                if is_actual_in_season:
                    unit_dict[actual_tz_date] = truth_value
        return loc_tz_date_to_actual_vals


    #
    # Score-related functions
    #

    def _update_model_score_changes(self):
        """
        Marks all my models' ModelScoreChange to now.
        """
        for forecast_model in self.models.all():
            forecast_model.score_change.update_changed_at()


#
# ---- Unit class ----
#

class Unit(models.Model):
    """
    Represents one of a project's units - just a string naming the target.
    """
    project = models.ForeignKey(Project, related_name='units', on_delete=models.CASCADE)
    name = models.TextField()


    def __repr__(self):
        return str((self.pk, self.name))


    def __str__(self):  # todo
        return basic_str(self)


#
# ---- TimeZero class ----
#

class TimeZero(models.Model):
    """
    A date that a target is relative to. Additionally, contains an optional data_version_date the specifies the database
    date at which models should work with for this timezero_date date. Akin to rolling back (versioning) the database
    to that date. Also contains optional season demarcation information in the form of a pair of fields, which are
    both required if a TimeZero marks a season start. The starting TimeZero includes that TimeZero (is inclusive).
     
    Assumes dates from any project can be converted to actual dates, e.g., from Dengue biweeks or CDC MMWR weeks
    ( https://ibis.health.state.nm.us/resource/MMWRWeekCalendar.html ).
    """
    project = models.ForeignKey(Project, related_name='timezeros', on_delete=models.CASCADE)
    timezero_date = models.DateField(help_text="A date that a target is relative to.")
    data_version_date = models.DateField(
        null=True, blank=True,
        help_text="The optional database date at which models should work with for the timezero_date.")  # nullable
    is_season_start = models.BooleanField(
        default=False,
        help_text="True if this TimeZero starts a season.")
    season_name = models.TextField(
        null=True, blank=True,
        max_length=50, help_text="The name of the season this TimeZero starts, if is_season_start.")  # nullable


    def __repr__(self):
        return str((self.pk, str(self.timezero_date), str(self.data_version_date),
                    self.is_season_start, self.season_name))


    def __str__(self):  # todo
        return basic_str(self)


    def save(self, *args, **kwargs):
        """
        Validates is_season_start and season_name.
        """
        if self.is_season_start and not self.season_name:
            raise ValidationError('passed is_season_start with no season_name')

        if not self.is_season_start and self.season_name:
            raise ValidationError('passed season_name but not is_season_start')

        # done
        super().save(*args, **kwargs)
