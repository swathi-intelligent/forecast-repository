from django.conf import settings
from django.conf.urls import url, include
from django.conf.urls.static import static
from django.contrib import admin
from rest_framework_jwt.views import obtain_jwt_token


urlpatterns = [
    # our app and REST API
    url(r'', include('forecast_app.urls')),
    url(r'^api/', include('forecast_app.api_urls')),
    url(r'^admin/', admin.site.urls),  # admin interface - mainly for User management
    url(r'^django-rq/', include('django_rq.urls')),
    url(r'^api-token-auth/', obtain_jwt_token, name='auth-jwt-get'),  # 'api-jwt-auth'
]

# use static() to add url mapping to serve static files during development (only)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# debug_toolbar
if settings.DEBUG:
    import debug_toolbar


    urlpatterns += url(r'^__debug__/', include(debug_toolbar.urls)),
