import click
import django


# set up django. must be done before loading models. NB: requires DJANGO_SETTINGS_MODULE to be set
django.setup()
from django.contrib.auth.models import User
from forecast_app.models import Project


@click.command()
def fix_owners_app():
    """
    App that sets project owners and model owners for the current programmatically-created projects. Used after making
    those projects and before uploading to Heroku remote postres database.
    """
    click.echo("fix_owners_app(): starting")

    # delete test projects
    click.echo("* deleting test projects")
    for project_name in ['public project', 'private project']:
        project = Project.objects.filter(name=project_name).first()
        if not project:
            click.echo("  x couldn't find project: '{}'".format(project_name))
            continue

        project.delete()
        click.echo("  - deleted project: {}".format(project))

    # delete test users
    click.echo("* deleting test users")
    for user_name in ['project_owner1', 'model_owner1']:
        user = User.objects.filter(username=user_name).first()
        if not user:
            click.echo("  x couldn't find test user '{}'".format(user_name))
            continue

        user.delete()
        click.echo("  - deleted user: {}".format(user))

    # change project and model owners
    for project_name, owner_name, model_owners in [
        ('CDC Flu challenge (2016-2017)', 'nick', ('cornell',)),
        ('Impetus Province Forecasts', 'nick', ('nectec', 'cornell',)),
        ('CDC FluSight ensemble (2017-2018)', 'nick', ('cornell',)),
    ]:
        click.echo("* project: '{}', '{}', {}".format(project_name, owner_name, model_owners))
        project = Project.objects.filter(name=project_name).first()
        if not project:
            click.echo("  x couldn't find project: '{}'".format(project_name))
            continue

        click.echo("** changing project owner: {}".format(owner_name))
        project_owner = User.objects.filter(username=owner_name).first()
        if not project_owner:
            click.echo("  x couldn't find project owner '{}'".format(project_owner))
            continue

        old_project_owner = project.owner
        project.owner = project_owner
        project.save()
        click.echo("  - changed project owner: {}: {} -> {}".format(project, old_project_owner, project.owner))

        click.echo("** changing model owners. from: {} to: {}".format(project.model_owners, model_owners))
        for model_owner_name in model_owners:
            model_owner = User.objects.filter(username=model_owner_name).first()
            if not model_owner:
                click.echo("  x couldn't find model owner: '{}'".format(model_owner))
                continue

            project.model_owners.add(model_owner)
        project.save()
        click.echo("  - added model owners: {} -> {}".format(project, project.model_owners.all()))

    # done
    click.echo("fix_owners_app(): done")


if __name__ == '__main__':
    fix_owners_app()