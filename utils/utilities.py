import logging

from django.template import Template, Context


logger = logging.getLogger(__name__)

#
# __str__()-related functions
#
from django.contrib.auth.models import User


def basic_str(obj):
    """
    Handy for writing quick and dirty __str__() implementations.
    """
    return obj.__class__.__name__ + ': ' + obj.__repr__()


#
# date formats and utilities
#

YYYY_MM_DD_DATE_FORMAT = '%Y-%m-%d'  # e.g., '2017-01-17'


def datetime_to_str(the_datetime):
    """
    Formats the_datetime (a datetime.datetime) using the Django date format pattern used in our templates:
    "Y-m-d h:i:s T". Note that Django's formats are from the PHP world, with different meanings of patterns. Elsewhere
    we use YYYY_MM_DD_DATE_FORMAT to format datetime.dates directly in python b/c it's the same as Django, but
    unfortunately Python doesn't have the "T" pattern needed to format datetime.datetime using the server's timezone.
    """
    message_template_str = """{{ the_datetime|date:"Y-m-d h:i:s T" }}"""
    message_template = Template(message_template_str)
    return message_template.render(Context({'the_datetime': the_datetime}))


#
# ---- User utilities ----
#

def get_or_create_super_po_mo_users(is_create_super):
    """
    A utility that creates (as necessary) three users - 'project_owner1', 'model_owner1', and a superuser. Should
    probably only be used for testing.

    :param is_create_super: boolean that controls whether a superuser is created. used only for testing b/c password is
        shown
    :return: a 4-tuple (if not create_super) or 6-tuple (if create_super) of Users and passwords:
        (superuser, superuser_password, po_user, po_user_password, mo_user, mo_user_password)
    """
    po_username = 'project_owner1'
    po_user_password = 'po1-asdf'
    po_user = User.objects.filter(username=po_username).first()
    if not po_user:
        logger.info("* creating PO user")
        po_user = User.objects.create_user(username=po_username, password=po_user_password)

    mo_username = 'model_owner1'
    mo_user_password = 'mo1-asdf'
    mo_user = User.objects.filter(username=mo_username).first()
    if not mo_user:
        logger.info("* creating MO user")
        mo_user = User.objects.create_user(username=mo_username, password=mo_user_password)

    super_username = 'superuser1'
    superuser_password = 'su1-asdf'
    superuser = User.objects.filter(username=super_username).first()
    if is_create_super and not superuser:
        logger.info("* creating supersuser")
        superuser = User.objects.create_superuser(username=super_username, password=superuser_password,
                                                  email='test@example.com')

    return (superuser, superuser_password, po_user, po_user_password, mo_user, mo_user_password) if is_create_super \
        else (po_user, po_user_password, mo_user, mo_user_password)
