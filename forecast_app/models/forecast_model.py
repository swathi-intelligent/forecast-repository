import csv

from django.db import connection
from django.db import models, transaction
from django.urls import reverse

import forecast_app.models.forecast  # we want Forecast, but import only the module to avoid circular imports
from forecast_app.models.project import Project
from utils.utilities import basic_str, filename_components, delphi_wili_for_epi_week


class ForecastModel(models.Model):
    """
    Represents a project's model entry by a competing team, including metadata, model-specific auxiliary data beyond
    core data, and a list of the actual forecasts.
    """
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True)

    name = models.CharField(max_length=200)

    description = models.CharField(max_length=2000, help_text="A few paragraphs describing the model. should include "
                                                              "information on reproducing the model’s results")

    url = models.URLField(help_text="The model's development URL")

    auxiliary_data = models.URLField(null=True,
                                     help_text="optional model-specific Zip file containing data files (e.g., CSV "
                                               "files) beyond Project.core_data that were used by the this model")

    def __repr__(self):
        return str((self.pk, self.name))

    def __str__(self):  # todo
        return basic_str(self)

    def get_absolute_url(self):
        return reverse('forecastmodel-detail', args=[str(self.id)])

    @transaction.atomic
    def load_forecast(self, csv_file_path, time_zero):  # faster alternative to ORM implementation using SQL
        """
        :param csv_file_path: Path to a CDC CSV forecast file
        :param time_zero: the TimeZero this forecast applies to
        :return: loads the data from the passed Path into my corresponding CDCData, and returns a new Forecast for it
        """
        forecast = forecast_app.models.forecast.Forecast.objects.create(
            forecast_model=self, time_zero=time_zero, data_filename=csv_file_path.name)

        # insert the data using direct SQL. for now simply use separate INSERTs per row
        with open(str(csv_file_path)) as csv_path_fp, \
                connection.cursor() as cursor:
            csv_reader = csv.reader(csv_path_fp, delimiter=',')
            next(csv_reader)  # skip header
            for row in csv_reader:  # might have 7 or 8 columns, depending on whether there's a trailing ',' in file
                location, target, row_type, unit, bin_start_incl, bin_end_notincl, value = row[:7]
                forecast.insert_data(cursor, location, target, row_type, unit,
                                     bin_start_incl, bin_end_notincl, value)
        # done
        return forecast

    def time_zero_for_timezero_date_str(self, timezero_date_str):
        """
        :return: the first TimeZero in forecast_model's Project that has a timezero_date matching timezero_date
        """
        for time_zero in self.project.timezero_set.all():
            if time_zero.timezero_date == timezero_date_str:
                return time_zero

        return None