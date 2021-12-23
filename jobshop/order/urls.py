# order/urls.py

from django.urls import path

from .views import *

urlpatterns = [
    path('', home, name='order.home'),
    path('list/', order_list_view, name='order.order_list_view'),
    path('list/search/', order_list_search, name='order.order_list_search'),
    path('fixed/schedule/', fixed_order, name='fixed'),
    path('csv/read/', order_read_csv, name='order.order_read_csv'),
    path('csv/delete/read/', order_delete_read_csv, name='order.order_delete_read_csv'),
    path('csv/create/', order_update_csv, name='order.csvCreate'),
    path('test/', order_test, name='test'),
]