# company/urls.py

from django.urls import path

from .views import *

urlpatterns = [
    path('', home, name='company.index'),
    path('product/regist/', comp_product_regist, name='company.product'),
    path('product/search/', comp_product_view, name='company.product.view'),
    path('facility/list/', comp_production_view, name='company.facility'),
    path('facility/list/edit/', fac_list_edit, name='company.fac_list_edit'),
    path('facility/list/all/', comp_facility_all, name='company.comp_facility_all'),
    path('info/list/', comp_list_view, name='company.list'),
    path('excel/create/', comp_update_excel, name='create'),
    path('csv/create/', comp_update_csv, name='csvCreate'),

    path('csv/comp/read/', comp_read_csv, name='company.comp_read_csv'),
    path('csv/comp/download/blank/', comp_csv_download_blank, name='company.comp_csv_download_blank'),
    path('csv/comp/download/', comp_csv_download, name='company.comp_csv_download'),
    path('avail/facility/', comp_avail_facility, name='company.comp_avail_facility'),

    path('csv/prod/read/', prod_read_csv, name='company.prod_read_csv'),
    path('csv/prod/upload/', comp_prod_update_csv, name='company.comp_prod_update_csv'),
    path('csv/prod/download/', prod_csv_download, name='company.prod_csv_download'),
    path('csv/prod/download/blank/', prod_csv_download_blank, name='company.prod_csv_download_blank'),

    path('csv/upload/', upload_file, name='upload'),
    path('delete/', delete, name='delete'),
]