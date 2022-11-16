
from django.contrib import admin
from django.urls import path
import predictor.views as v
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', v.index, name='index'),
    path('predict/', v.predict, name='predict'),
]

urlpatterns += staticfiles_urlpatterns()
