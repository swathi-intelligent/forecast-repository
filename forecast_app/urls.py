from django.conf.urls import url, include

from . import views


urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^about$', views.about, name='about'),

    url(r'^accounts/', include('django.contrib.auth.urls')),

    url(r'^project/(?P<pk>\d+)$', views.ProjectDetailView.as_view(), name='project-detail'),
    url(r'^project/(?P<project_pk>\d+)/visualizations$', views.project_visualizations, name='project-visualizations'),
    url(r'^project/(?P<project_pk>\d+)/template$', views.template_detail, name='template-data-detail'),
    url(r'^project/(?P<model_with_cdc_data_pk>\d+)/json', views.download_json_for_model_with_cdc_data,
        {'type': 'project'},
        name='download-template-json'),

    url(r'^model/(?P<pk>\d+)$', views.ForecastModelDetailView.as_view(), name='forecastmodel-detail'),

    url(r'^user/(?P<pk>\d+)$', views.UserDetailView.as_view(), name='user-detail'),

    url(r'^forecast/(?P<pk>\d+)$', views.ForecastDetailView.as_view(), name='forecast-detail'),
    url(r'^forecast/(?P<model_with_cdc_data_pk>\d+)/json', views.download_json_for_model_with_cdc_data,
        {'type': 'forecast'},
        name='download-forecast-json'),
    url(r'^forecast/(?P<forecast_pk>\d+)/delete$', views.delete_forecast, name='delete-forecast'),
    url(r'^forecast/(?P<forecast_model_pk>\d+)/upload/(?P<timezero_pk>\d+)$', views.upload_forecast,
        name='upload-forecast'),

]
