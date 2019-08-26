from collections import defaultdict
from itertools import groupby

from django.db import connection

from forecast_app.models import TruthData, TimeZero, Forecast, ForecastModel
from forecast_app.models.project import TargetBinLwr, Target
from forecast_app.scores.definitions import _validate_score_targets_and_data, logger


def _calc_bin_score(score, forecast_model, save_score_fcn, **kwargs):
    """
    Function shared by log and pit scores.

    :param: save_score_fcn: a function that creates and saves a ScoreValue. args: see save_score_fcn() call below
    :param: kwargs: passed through to save_score_fcn
    """
    try:
        _validate_score_targets_and_data(forecast_model)
    except RuntimeError as rte:
        logger.warning(rte)
        return

    # collect errors so we don't log thousands of duplicate messages. dict format:
    #   {(timezero_pk, location_pk, target_pk): count, ...}:
    # note that the granularity is poor - there are multiple possible errors related to a particular 3-tuple
    tz_loc_targ_pks_to_error_count = defaultdict(int)  # helps eliminate duplicate warnings

    # cache the three necessary bins and values - lwrs, truth, and forecasts
    # 1/3 binlwrs: [target_pk] -> [lwr_1, ...]:
    targ_pk_to_bin_lwrs = _targ_pk_to_bin_lwrs(forecast_model.project)

    # 2/3 truth: [timezero_pk][location_pk][target_pk] -> true_bin_lwr:
    tz_loc_targ_pk_to_true_bin_lwr = _tz_loc_targ_pk_to_true_bin_lwr(forecast_model.project)

    # 3/3 forecast: [timezero_pk][location_pk][target_pk][bin_lwr] -> predicted_value:
    tz_loc_targ_pk_bin_lwr_to_pred_val = _tz_loc_targ_pk_bin_lwr_to_pred_val(forecast_model)

    # it is convenient to iterate over truths to get all timezero/location/target combinations. this will omit forecasts
    # with no truth, but that's OK b/c without truth, a forecast makes no contribution to the score. we use direct SQL
    # to work with PKs and avoid ORM object lookup overhead, mainly for TruthData -> TimeZero -> Forecast -> PK
    for time_zero_pk, forecast_pk, location_pk, target_pk, truth_value in \
            _truth_data_pks_for_forecast_model(forecast_model):
        # get binlwrs for this forecast
        try:
            bin_lwrs = targ_pk_to_bin_lwrs[target_pk]
        except KeyError:
            error_key = (time_zero_pk, location_pk, target_pk)
            tz_loc_targ_pks_to_error_count[error_key] += 1
            continue  # skip this forecast's contribution to the score

        # get and validate truth for this forecast
        try:
            true_bin_lwr = tz_loc_targ_pk_to_true_bin_lwr[time_zero_pk][location_pk][target_pk]
            true_bin_idx = bin_lwrs.index(true_bin_lwr)  # NB: non-deterministic for (None, None) true bin keys!
        except (KeyError, ValueError):
            error_key = (time_zero_pk, location_pk, target_pk)
            tz_loc_targ_pks_to_error_count[error_key] += 1
            continue  # skip this forecast's contribution to the score

        # get forecast bins and predicted values for this forecast
        try:
            bin_lwr_to_pred_val = tz_loc_targ_pk_bin_lwr_to_pred_val[time_zero_pk][location_pk][target_pk]
        except KeyError:
            error_key = (time_zero_pk, location_pk, target_pk)
            tz_loc_targ_pks_to_error_count[error_key] += 1
            continue  # skip this forecast's contribution to the score

        # dispatch to scoring function
        save_score_fcn(score, forecast_pk, location_pk, target_pk, truth_value, bin_lwrs,
                       bin_lwr_to_pred_val, true_bin_lwr, true_bin_idx, **kwargs)

    # print errors
    for (timezero_pk, location_pk, target_pk) in sorted(tz_loc_targ_pks_to_error_count.keys()):
        count = tz_loc_targ_pks_to_error_count[timezero_pk, location_pk, target_pk]
        logger.warning(f"_calculate_pit_score_values(): missing {count} truth value(s): "
                       "timezero_pk={timezero_pk}, location_pk={location_pk}, target_pk={target_pk}")


def _truth_data_pks_for_forecast_model(forecast_model):
    sql = f"""
        SELECT td.time_zero_id, f.id, td.location_id, td.target_id, td.value
        FROM {TruthData._meta.db_table} AS td
               LEFT JOIN {TimeZero._meta.db_table} AS tz ON td.time_zero_id = tz.id
               LEFT JOIN {Forecast._meta.db_table} AS f ON tz.id = f.time_zero_id
               LEFT JOIN {ForecastModel._meta.db_table} AS fm ON f.forecast_model_id = fm.id
        WHERE fm.id = %s;
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, (forecast_model.pk,))
        return cursor.fetchall()


#
# ---- predictive distribution (aka 'bin') lookup functions ----
#

def _tz_loc_targ_pk_to_true_bin_lwr(project):
    """
    Returns project's TruthData merged with the project's BinLwrs:

        [timezero_pk][location_pk][target_pk] -> true_bin_lwr

    We need the TargetBinLwr to get lwr and upper for the truth.
    """
    sql = f"""
        SELECT truthd.time_zero_id, truthd.location_id, truthd.target_id, tblwr.lwr
        FROM {TruthData._meta.db_table} as truthd
               LEFT JOIN {TargetBinLwr._meta.db_table} as tblwr
                    ON truthd.target_id = tblwr.target_id
               LEFT JOIN {Target._meta.db_table} as t
                    ON tblwr.target_id = t.id
        WHERE t.project_id = %s
          AND ((truthd.value >= tblwr.lwr) OR ((truthd.value IS NULL) AND (tblwr.lwr IS NULL)))
          AND ((truthd.value < tblwr.upper) OR ((truthd.value IS NULL) AND (tblwr.upper IS NULL)))
        ORDER BY truthd.time_zero_id, truthd.location_id, truthd.target_id, tblwr.lwr
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, (project.pk,))
        rows = cursor.fetchall()

    # build the dict
    tz_loc_targ_pks_to_true_bin_lwr = {}  # {timezero_pk: {location_pk: {target_id: true_bin_lwr}}}
    for time_zero_id, loc_target_val_grouper in groupby(rows, key=lambda _: _[0]):
        loc_targ_pks_to_truth_bin_start = {}  # {location_pk: {target_id: true_bin_lwr}}
        tz_loc_targ_pks_to_true_bin_lwr[time_zero_id] = loc_targ_pks_to_truth_bin_start
        for location_id, target_val_grouper in groupby(loc_target_val_grouper, key=lambda _: _[1]):
            target_pk_to_truth = {}  # {target_id: true_bin_lwr}
            loc_targ_pks_to_truth_bin_start[location_id] = target_pk_to_truth
            for _, _, target_id, true_bin_lwr in target_val_grouper:
                target_pk_to_truth[target_id] = true_bin_lwr

    return tz_loc_targ_pks_to_true_bin_lwr


def _targ_pk_to_bin_lwrs(project):
    """
    Returns project's lwr data as a dict: [target_pk] -> [lwr_1, ...]. Each list is sorted by lwr.
    Only returns rows whose targets match non_date_targets().
    """
    targets = project.non_date_targets()
    target_bin_lwr_qs = TargetBinLwr.objects \
        .filter(target__in=targets) \
        .order_by('target__id', 'lwr') \
        .values_list('target__id', 'lwr')

    # build the dict
    target_pk_to_bin_lwrs = {}  # {target_id: [bin_lwr_1, ...]}
    for target_id, lwr_grouper in groupby(target_bin_lwr_qs, key=lambda _: _[0]):
        target_pk_to_bin_lwrs[target_id] = [lwr for _, lwr in lwr_grouper]

    return target_pk_to_bin_lwrs


def _tz_loc_targ_pk_bin_lwr_to_pred_val(forecast_model):
    """
    Returns prediction data for all forecasts in forecast_model as a dict:

        [timezero_pk][location_pk][target_pk][bin_lwr] -> predicted_value

    Only returns rows whose targets match non_date_targets().
    """
    targets = forecast_model.project.non_date_targets()
    forecast_data_qs = ForecastData.objects \
        .filter(forecast__forecast_model=forecast_model,
                is_point_row=False,
                target__in=targets) \
        .order_by('forecast__time_zero__id', 'location__id', 'target__id') \
        .values_list('forecast__time_zero__id', 'location__id', 'target__id', 'bin_lwr', 'value')

    # build the dict: {timezero_pk: {location_pk: {target_id: {bin_lwr_1: predicted_value_1, ...}}}}:
    tzltpk_to_forec_st_to_pred_val = {}
    for time_zero_id, loc_target_val_grouper in groupby(forecast_data_qs, key=lambda _: _[0]):
        ltpk_to_forec_start_to_pred_val = {}  # {location_pk: {target_id: {bin_lwr_1: predicted_value_1, ...}}}
        tzltpk_to_forec_st_to_pred_val[time_zero_id] = ltpk_to_forec_start_to_pred_val
        for location_id, target_val_grouper in groupby(loc_target_val_grouper, key=lambda _: _[1]):
            # {target_id: {bin_lwr_1: predicted_value_1, ...}}:
            tpk_to_forec_start_to_pred_val = defaultdict(dict)
            ltpk_to_forec_start_to_pred_val[location_id] = tpk_to_forec_start_to_pred_val
            for _, _, target_id, bin_lwr, pred_value in target_val_grouper:
                tpk_to_forec_start_to_pred_val[target_id][bin_lwr] = pred_value

    return tzltpk_to_forec_st_to_pred_val
