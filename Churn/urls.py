from django.contrib import admin
from admins import views as admins
from django.urls import path
from users import views as usr
from . import views as mainView
from django.contrib.staticfiles.urls import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf import settings

urlpatterns = [
    path('', mainView.index, name='index'),
    path('index/', mainView.index, name='index'),
    path('login/', mainView.login_view, name='login'),
    path('logout/', mainView.logout, name='logout'),

    ### Global Auth Actions
    path("UserRegisterActions/", usr.UserRegisterActions, name="UserRegisterActions"),
    path("UserLoginCheck/", usr.UserLoginCheck, name="UserLoginCheck"),

    ### User/Admin Shared Analytics Views
    path("UserHome/", usr.UserHome, name="UserHome"),
    path("UserProfile/", usr.UserProfile, name="UserProfile"),
    path("training/", usr.training, name="training"),
    path("prediction/", usr.prediction, name="prediction"),
    path("live_prediction/", usr.live_prediction, name="live_prediction"),
    path("DatasetView/", usr.DatasetView, name="DatasetView"),

    ### Admin Only Management Views
    path("AdminHome/", admins.AdminHome, name="AdminHome"),
    path("ViewRegisteredUsers/", admins.ViewRegisteredUsers, name="ViewRegisteredUsers"),
    path("AdminDeleteUser/", admins.AdminDeleteUser, name="AdminDeleteUser"),
    path("AdminUpdateUser/", admins.AdminUpdateUser, name="AdminUpdateUser"),
    path("generate_report/", usr.generate_report, name="generate_report"),
    path("admin-report/", admins.AdminReport, name="AdminReport"),
    path("admin-report/download/", admins.AdminDownloadReport, name="AdminDownloadReport"),
]
urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)