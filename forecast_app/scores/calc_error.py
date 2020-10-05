import logging
from itertools import groupby
import pandas as pd

from forecast_app.models import PointPrediction
from django.db import connection

logger = logging.getLogger(__name__)


def _calculate_error_score_values(score, forecast_model, is_absolute_error):
    """
    Implements the 'error' and 'abs_error' scores. Creates ScoreValue instances for the passed args, saving them into
    the passed score. The score is simply `true_value - predicted_value` (optionally passed to abs() based on
    is_absolute_error) for each combination of Unit + Target in forecast_model's project. Runs in the calling thread
    and therefore blocks. Note that this implementation uses a naive approach to calculating scores, iterating over
    truth and forecast tables instead of caching.

    :param score: a Score
    :param forecast_model: a ForecastModel
    :param is_absolute_error: True if abs() should be called
    """
    # for now, call our new dataframe-based API on each of forecast_model's forecasts, i.e., the new granularity.
    #
    # proposed API:
    # - input: score calculation function is called with these args:
    #          score, forecast, truth_dataframe, predictions_dataframe
    # - output: score_values_dataframe
    #
    # ** simulate the top-level update_score_for_model() -> update_score_for_forecast() **
    for forecast in forecast_model.forecasts.all():
        truth_dataframe = _truth_dataframe_for_forecast(forecast)
        predictions_dataframe = _predictions_dataframe_for_forecast(forecast)
        score_values_dataframe = _calculate_error_score_values_df(truth_dataframe, predictions_dataframe,
                                                                  is_absolute_error)
        _insert_score_values_dataframe(score_values_dataframe)


def _truth_dataframe_for_forecast(forecast):
    project = forecast.forecast_model.project
    truth_qs = project\
        .truth_data_qs()\
        .filter(time_zero__timezero_date=forecast.time_zero.timezero_date)\
        .values_list('time_zero__id', 'unit__id', 'target__id', 'value_i', 'value_f', 'value_t', 'value_d', 'value_b')\
        .iterator()
    # print('xx', list(truth_qs))
    truth_dataframe = pd.DataFrame(
        data=truth_qs,
        columns=['timezero_id', 'unit_id', 'target_id', 'value_i', 'value_f', 'value_t', 'value_d', 'value_b']
    )
    truth_dataframe['value'] = truth_dataframe.bfill(axis=1).loc[:, 'value_i'].apply(str)
    truth_dataframe = truth_dataframe.loc[:, ['timezero_id', 'unit_id', 'target_id', 'value']]

    print('xx', truth_dataframe)

    return truth_dataframe


def _predictions_dataframe_for_forecast(forecast):
    pass  # todo xx


# ** API function for error and abs_error calculations **
def _calculate_error_score_values_df(truth_dataframe, predictions_dataframe, is_absolute_error):
    pass  # todo xx


def _insert_score_values_dataframe(score_values_dataframe):
    pass  # todo xx


#
# OLD
#

def _calculate_error_score_values_old(score, forecast_model, is_absolute_error):
    from forecast_app.scores.bin_utils import _insert_score_values  # avoid circular imports
    from forecast_app.scores.definitions import _validate_score_targets_and_data

    try:
        targets = _validate_score_targets_and_data(forecast_model)
    except RuntimeError as rte:
        logger.warning(f"_calculate_error_score_values(): _validate_score_targets_and_data() failed. "
                       f"rte={rte!r}, score={score}, forecast_model={forecast_model}")
        return

    # step 1/2: build tz_unit_targ_pk_to_pt_pred_value: [timezero_id][unit_id][target_id] -> point_value
    tz_unit_targ_pk_to_pt_pred_value = {}
    point_predictions_qs = PointPrediction.objects \
        .filter(forecast__forecast_model=forecast_model, target__in=targets) \
        .order_by('forecast__time_zero__id', 'unit__id', 'target__id') \
        .values_list('forecast__time_zero__id', 'unit__id', 'target__id',
                     'value_i', 'value_f', 'value_t', 'value_d', 'value_b')  # only one of value_* is non-None
    for timezero_id, unit_target_val_grouper in groupby(point_predictions_qs, key=lambda _: _[0]):
        tz_unit_targ_pk_to_pt_pred_value[timezero_id] = {}
        for unit_id, target_val_grouper in groupby(unit_target_val_grouper, key=lambda _: _[1]):
            tz_unit_targ_pk_to_pt_pred_value[timezero_id][unit_id] = {}
            for _, _, target_id, value_i, value_f, value_t, value_d, value_b in target_val_grouper:
                value = PointPrediction.first_non_none_value(value_i, value_f, None, value_d, None)
                tz_unit_targ_pk_to_pt_pred_value[timezero_id][unit_id][target_id] = value

    # step 2/2: iterate over truths, calculating scores. it is convenient to iterate over truths to get all
    # timezero/unit/target combinations. this will omit forecasts with no truth, but that's OK b/c without truth, a
    # forecast makes no contribution to the score. note that we collect all ScoreValue rows and then bulk insert them as
    # an optimization, rather than create separate ORM instances
    score_values = []  # list of 5-tuples: (score.pk, forecast.pk, unit.pk, target.pk, score_value)
    timezero_id_to_forecast_id = {forecast.time_zero.pk: forecast.pk for forecast in forecast_model.forecasts.all()}
    truth_data_qs = forecast_model.project.truth_data_qs() \
        .filter(target__in=targets) \
        .values_list('time_zero__id', 'unit__id', 'target__id',
                     'value_i', 'value_f', 'value_t', 'value_d', 'value_b')  # only one of value_* is non-None
    num_warnings = 0
    for timezero_id, unit_id, target_id, value_i, value_f, value_t, value_d, value_b in truth_data_qs:
        truth_value = PointPrediction.first_non_none_value(value_i, value_f, value_t, value_d, value_b)
        if truth_value is None:
            num_warnings += 1
            continue  # skip this timezero's contribution to the score
        try:
            predicted_value = tz_unit_targ_pk_to_pt_pred_value[timezero_id][unit_id][target_id]
            score_value = abs(truth_value - predicted_value) if is_absolute_error else truth_value - predicted_value
            score_values.append((score.pk, timezero_id_to_forecast_id[timezero_id], unit_id, target_id, score_value))
        except KeyError as ke:  # no predicted value for one of timezero_id, unit_id, target_id
            num_warnings += 1
            continue  # skip this timezero's contribution to the score

    # insert the ScoreValues!
    _insert_score_values(score_values)

    # print warning count
    logger.warning(f"_calculate_error_score_values(): done. score={score}, forecast_model={forecast_model}, "
                   f"num_warnings={num_warnings}")
