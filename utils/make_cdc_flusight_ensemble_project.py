import timeit
from collections import defaultdict
from pathlib import Path

import click
import django
import pymmwr
import yaml
from django.template import Template, Context


# set up django. must be done before loading models. NB: requires DJANGO_SETTINGS_MODULE to be set
django.setup()

from utils.utilities import cdc_csv_components_from_data_dir, cdc_csv_filename_components, season_start_year_for_date
from forecast_app.models import Project, ForecastModel, TimeZero
from utils.make_cdc_flu_challenge_project import get_or_create_super_po_mo_users, create_targets
from utils.cdc import CDC_CONFIG_DICT


@click.command()
@click.argument('component_models_dir', type=click.Path(file_okay=False, exists=True))
@click.option('--make_project', is_flag=True, default=False)
@click.option('--load_data', is_flag=True, default=False)
def make_cdc_flusight_ensemble_project_app(component_models_dir, make_project, load_data):
    """
    Manages creating a Project for the https://github.com/FluSightNetwork/cdc-flusight-ensemble project and loading its
    models, based on the two flags.

    If make_project: Creates the Project (deleting existing if exists!), user group, and two classes of users, along
    with creating (but not necessarily loading data for) the Models found in component_models_dir, using the
    metadata.txt files.

    If load_data: Loads data from the models in component_models_dir. Errors if make_project was not done previously.

    :param: component_models_dir: a directory cloned from
        https://github.com/FluSightNetwork/cdc-flusight-ensemble/tree/master/model-forecasts/component-models , which
        has then been normalized via normalize_cdc_flusight_ensemble_filenames_app.py .
    """
    start_time = timeit.default_timer()
    component_models_dir = Path(component_models_dir)
    click.echo("* make_cdc_flusight_ensemble_project_app(): component_models_dir={}, make_project={}, load_data={}"
               .format(component_models_dir, make_project, load_data))

    project_name = 'CDC FluSight ensemble (2017-2018)'
    project = Project.objects.filter(name=project_name).first()  # None if doesn't exist
    template_52 = Path('forecast_app/tests/2016-2017_submission_template.csv')  # todo xx move into repo
    template_53 = Path('forecast_app/tests/2016-2017_submission_template-plus-bin-53.csv')  # ""
    if make_project:
        if project:
            click.echo("* Deleting existing project: {}".format(project))
            project.delete()

        # create the Project (and Users if necessary)
        po_user, _, mo_user, _ = get_or_create_super_po_mo_users(create_super=False)
        project = Project.objects.create(
            owner=po_user,
            is_public=True,
            name=project_name,
            description="Guidelines and forecasts for a collaborative U.S. influenza forecasting project. "
                        "http://flusightnetwork.io/",
            home_url='https://github.com/FluSightNetwork/cdc-flusight-ensemble',
            core_data='https://github.com/FluSightNetwork/cdc-flusight-ensemble/tree/master/model-forecasts/component-models',
            config_dict=CDC_CONFIG_DICT)
        project.model_owners.add(mo_user)
        project.save()
        click.echo("* Created project: {}".format(project))

        # load the template. NB: this project is different from others in that there are two templates that apply, based
        # on the season/year: some have 53 days, which means the template being validated against for that year must
        # have a bin for week 53. we handle this using two templates:
        #
        # - 2016-2017_submission_template.csv: last bin is 52,53
        # - 2016-2017_submission_template-plus-bin-53.csv: "" 53,54
        #
        # because projects can only have one template, we arbitrarily choose the former. HOWEVER, this means future
        # forecast validation will fail if it's for a year with 53 days. for reference, we use
        # pymmwr.mmwr_weeks_in_year() determine the number of weeks in a year
        click.echo("  loading template")
        project.load_template(template_52)

        targets = create_targets(project)
        click.echo("  created {} Targets: {}".format(len(targets), targets))

        # create TimeZeros. we use an arbitrary model's *.cdc.csv files to get them (all models should have same ones,
        # which is checked during forecast validation later)
        time_zeros = create_timezeros(project, first_subdirectory(component_models_dir))  # assumes no non-model subdirs
        click.echo("  created {} TimeZeros: {}".format(len(time_zeros), time_zeros))

        click.echo("* Creating models")
        models = make_cdc_flusight_ensemble_models(project, component_models_dir, po_user)
        click.echo("  created {} model(s): {}".format(len(models), models))
    elif not project:  # not make_project, but couldn't find existing
        raise RuntimeError("Could not find existing project named '{}'".format(project_name))

    if load_data:
        click.echo("* Loading forecasts")
        model_name_to_forecasts = load_cdc_flusight_ensemble_forecasts(project, component_models_dir,
                                                                       template_52, template_53)
        click.echo("  loaded {} forecast(s)".format(sum(map(len, model_name_to_forecasts.values()))))

    click.echo("* Done. time: {}".format(timeit.default_timer() - start_time))


def first_subdirectory(directory):
    for subdir in directory.iterdir():
        if subdir.is_dir():
            return subdir

    return None


def create_timezeros(project, model_dir):
    """
    Create TimeZeros for project based on model_dir. Returns a list of them.
    """
    time_zeros = []
    for _, time_zero, _, data_version_date in cdc_csv_components_from_data_dir(model_dir):
        time_zeros.append(TimeZero.objects.create(project=project,
                                                  timezero_date=time_zero,
                                                  data_version_date=data_version_date))
    return time_zeros


def make_cdc_flusight_ensemble_models(project, component_models_dir, model_owner):
    """
    Creates ForecastModels for project based on the directories under component_models_dir, with model_owner as the
    owner for all of them.
    """
    models = []
    for model_dir in component_models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # get model name and description from metadata.txt
        metadata_dict = metadata_dict_for_file(model_dir / 'metadata.txt')
        model_name = metadata_dict['model_name']

        # build description
        description_template_str = """<em>Team name</em>: {{ team_name }}.
        <em>Team members</em>: {{ team_members }}.
        <em>Data source(s)</em>: {% if data_source1 %}{{ data_source1 }}{% if data_source2 %}, {{ data_source2 }}{% endif %}{% else %}None specified{% endif %}.
        <em>Methods</em>: {{ methods }}
        """
        description_template = Template(description_template_str)
        description = description_template.render(
            Context({'team_name': metadata_dict['team_name'],
                     'team_members': metadata_dict['team_members'],
                     'data_source1': metadata_dict['data_source1'] if 'data_source1' in metadata_dict else None,
                     'data_source2': metadata_dict['data_source2'] if 'data_source2' in metadata_dict else None,
                     'methods': metadata_dict['methods'],
                     }))

        home_url = 'https://github.com/FluSightNetwork/cdc-flusight-ensemble/tree/master/model-forecasts/component-models' \
                   + '/' + model_dir.name
        forecast_model = ForecastModel.objects.create(owner=model_owner, project=project, name=model_name,
                                                      description=description, home_url=home_url)
        models.append(forecast_model)
    return models


def metadata_dict_for_file(metadata_file):
    with open(metadata_file) as metadata_fp:
        metadata_dict = yaml.safe_load(metadata_fp)
    return metadata_dict


def load_cdc_flusight_ensemble_forecasts(project, component_models_dir, template_52, template_53):
    """
    Loads forecast data for all models corresponding to directories under component_models_dir. Assumes model names
    in each directory's metadata.txt matches those in project, as done by make_cdc_flusight_ensemble_models(). see above
    note re: the two templates.
    """
    model_name_to_forecasts = defaultdict(list)
    for model_dir in component_models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        click.echo("** {}".format(model_dir))
        metadata_dict = metadata_dict_for_file(model_dir / 'metadata.txt')
        model_name = metadata_dict['model_name']
        forecast_model = project.models.filter(name=model_name).first()
        if not forecast_model:
            raise RuntimeError("Couldn't find model named '{}' in project {}".format(model_name, project))


        def time_zero_to_template(time_zero):
            season_start_year = season_start_year_for_date(time_zero.timezero_date)
            return {52: template_52, 53: template_53}[pymmwr.mmwr_weeks_in_year(season_start_year)]


        def is_load_file(cdc_csv_file):
            # only accept EW43 through EW18 per: "Following CDC guidelines from 2017/2018 season, using scores from
            # files from each season labeled EW43 through EW18 (i.e. files outside that range will not be considered)"
            time_zero, _, _ = cdc_csv_filename_components(cdc_csv_file.name)
            ywd_mmwr_dict = pymmwr.date_to_mmwr_week(time_zero)
            mmwr_week = ywd_mmwr_dict['week']
            return (mmwr_week <= 18) or (mmwr_week >= 43)


        def callback(cdc_csv_file, reason, exception):
            if reason == 'ok':
                click.echo("o\t{}\t".format(cdc_csv_file.name)),
            elif reason == 'skip':
                click.echo("s\t{}\t".format(cdc_csv_file.name)),
            else:  # 'fail'
                click.echo("f\t{}\t{}".format(cdc_csv_file.name, exception)),


        forecasts = forecast_model.load_forecasts_from_dir(
            model_dir,
            time_zero_to_template=time_zero_to_template,
            is_load_file=is_load_file,
            callback=callback)
        model_name_to_forecasts[model_name].extend(forecasts)

    return model_name_to_forecasts


if __name__ == '__main__':
    make_cdc_flusight_ensemble_project_app()