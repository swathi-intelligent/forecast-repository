import logging

import django_rq
from django.db import transaction
from django.shortcuts import get_object_or_404

from forecast_app.models import Score, ScoreValue, Project
from utils.mean_absolute_error import _model_id_to_point_values_dict, _model_id_to_forecast_id_tz_dates, \
    iterate_forecast_errors


logger = logging.getLogger(__name__)

#
# ---- Absolute Error score functions ----
#

# todo xx this is a place-holder implementation. the design needs fleshing out so that this can be a method on a Score subcass, for example

ABSOLUTE_ERROR_SCORE_NAME = 'Absolute Error'  # official name
ABSOLUTE_ERROR_SCORE_DESCRIPTION = "The absolute of error between the model's point estimate and the truth value. " \
                                   "Lower is better."  # official description


@transaction.atomic
def calculate_absolute_error_score_values(project):
    """
    The top-level function that creates ScoreValue instances for the passed args, saving them into the passed score.
    Calculates scores for all (!) ForecastModels in project. Uses ABSOLUTE_ERROR_SCORE_NAME to look up the Score
    instance to store the score values into.

    :param project: a Project
    """
    # clear or create the Score itself, deleting any existing ScoreValues if any
    abs_err_score, is_created = Score.objects.get_or_create(name=ABSOLUTE_ERROR_SCORE_NAME,
                                                            description=ABSOLUTE_ERROR_SCORE_DESCRIPTION)
    if not is_created:
        abs_err_score.values.filter(forecast__forecast_model__project=project).delete()

    logger.debug("calculate_absolute_error_score_values(): entered. project={}, score={}"
                 .format(project, abs_err_score))

    # validate targets
    targets = project.visualization_targets()
    if not targets:
        logger.warning("calculate_absolute_error_score_values(): no visualization targets. project={}".format(project))
        return

    # cache all the data we need for all models
    locations = project.locations.all()
    model_id_to_point_values_dict = _model_id_to_point_values_dict(project, [target.name for target in targets])
    model_id_to_forecast_id_tz_dates = _model_id_to_forecast_id_tz_dates(project)
    loc_target_tz_date_to_truth = project.location_target_name_tz_date_to_truth()

    # calculate for all combinations of model, location, and target
    for forecast_model in project.models.order_by('name'):
        if not forecast_model.forecasts.exists():
            logger.warning("calculate_absolute_error_score_values(): could not calculate absolute errors: model had "
                           "no data: {}".format(forecast_model))
            continue

        for location in locations:
            for target in targets:
                forecast_to_point_dict = model_id_to_point_values_dict[forecast_model.pk] \
                    if forecast_model.pk in model_id_to_point_values_dict \
                    else {}
                forecast_id_tz_dates = model_id_to_forecast_id_tz_dates[forecast_model.pk] \
                    if forecast_model.pk in model_id_to_forecast_id_tz_dates \
                    else {}
                calculate_absolute_error(abs_err_score, forecast_model, location, target,
                                         forecast_to_point_dict, forecast_id_tz_dates, loc_target_tz_date_to_truth)

    # done
    abs_err_score.set_last_update_for_project(project)
    logger.debug("calculate_absolute_error_score_values(): done. # values={}".format(
        ScoreValue.objects.filter(score=abs_err_score, forecast__forecast_model__project=project).count()))


def calculate_absolute_error(abs_err_score, forecast_model, location, target,
                             forecast_to_point_dict, forecast_id_tz_dates, loc_target_tz_date_to_truth):
    def score_value_error_fcn(forecast_id, forecast_timezero_date, predicted_value, true_value):
        ScoreValue.objects.create(forecast_id=forecast_id, location=location, target=target,
                                  score=abs_err_score, value=(abs(true_value - predicted_value)))


    iterate_forecast_errors(forecast_model, location.name, target.name,
                            forecast_to_point_dict, forecast_id_tz_dates, loc_target_tz_date_to_truth,
                            score_value_error_fcn)


#
# RQ-related queueing functions
#

def enqueue_score_updates_all_projs():
    for project in Project.objects.all():
        logger.debug("enqueuing update project scores. project={}".format(project))
        django_rq.enqueue(_update_project_scores, project.pk)


def _update_project_scores(project_pk):
    """
    Enqueue helper function.
    """
    project = get_object_or_404(Project, pk=project_pk)
    # see: [todo xx this is a place-holder implementation. the design needs fleshing out so that this can be a method on a Score subcass, for example]
    calculate_absolute_error_score_values(project)