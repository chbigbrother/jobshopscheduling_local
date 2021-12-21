# company/urls.py

from django.urls import path

from .views import *

urlpatterns = [
    path('', home, name='company.index'),
    path('daily/production/', comp_daily_view, name='company.daily'),
    path('facility/list/', comp_facility_view, name='company.facility'),
    path('info/list/', comp_list_view, name='company.list'),
    path('excel/create/', comp_update_excel, name='create'),
    path('csv/create/', comp_update_csv, name='csvCreate'),
    path('csv/upload/', upload_file, name='upload'),
    path('delete/', delete, name='delete'),
    # path('create/', ArticleCreate.as_view(), name='create'),
    # path('<int:pk>/', ArticleDetail.as_view(), name='detail'),
    # path('<int:pk>/update/', ArticleUpdate.as_view(), name='update'),
    # path('<int:pk>/delete/', ArticleDelete.as_view(), name='delete'),
]