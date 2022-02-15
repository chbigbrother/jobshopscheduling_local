# order/urls.py

from django.urls import path

from .views import *
app_name = 'order'
urlpatterns = [
    path('', home, name='order.home'),
    path('list/', order_list_view, name='order.order_list_view'),
    path('list/search/', order_list_search, name='order.order_list_search'),
    path('list/search/edit/', order_list_edit, name='order.order_list_edit'),
    path('list/search/delete/', order_list_delete, name='order.order_list_delete'),
    path('fixed/schedule/', fixed_order, name='fixed'),
    path('csv/download/blank/', order_csv_download_blank, name='order.order_csv_download_blank'),
    path('csv/read/', order_read_csv, name='order.order_read_csv'),
    path('csv/delete/read/', order_delete_read_csv, name='order.order_delete_read_csv'),
    path('csv/create/', order_update_csv, name='order.csvCreate'),
    path('modal/create/', order_update_modal, name='order.order_update_modal'),
    path('test/', order_test, name='test'),
]