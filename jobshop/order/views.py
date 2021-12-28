#-*- coding: utf-8 -*-
from django.urls import reverse_lazy
from django.http import HttpResponse, JsonResponse
from django.views.generic import ListView, CreateView, DetailView, UpdateView, DeleteView, TemplateView
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.models import User, Group
from django.db.models import Max
from django.template import loader
from rest_framework import serializers
from openpyxl import load_workbook
from fileutils.forms import FileUploadCsv
from .models import OrderSchedule, OrderList
from company.models import *
from common.views import id_generate
from datetime import timedelta
from .models import *
import datetime, json, csv
import pandas as pd

# 수주관리등록 order management register HTML
def home(request):
    template_name = 'order/orderregist.html'
    context = {
        'path': '수주관리 / 수주관리등록',
        'selected': 'ordermanagement'
    }
    return render(request, template_name, context)

# 수주관리검색 order management search HTML
def order_list_view(request):
    template_name = 'order/orderlist.html'

    dateFrom, dateTo, order_list = order_list_query(request)
    context = {
        'dateFrom': dateFrom,
        'dateTo': dateTo,
        'order_list': order_list,
        'path': '수주관리 / 수주관리검색',
        'selected': 'ordermanagement'
    }
    return render(request, template_name, context)

def order_list_query(request):
    date = datetime.datetime.today() - timedelta(days=3)

    if 'dateFrom' in request.GET:
        sch_date_from = datetime.datetime.strptime(request.GET['dateFrom'], "%Y-%m-%d")
        sch_date_to = datetime.datetime.strptime(request.GET['dateTo'], "%Y-%m-%d")

        date_from = request.GET['dateFrom'].replace('-', '')
        date_to = request.GET['dateTo'].replace('-', '')

        order_list = OrderList.objects.filter(sch_date__gte=date_from).filter(sch_date__lte=date_to)
    else:
        sch_date_from = date
        sch_date_to = datetime.datetime.today()
        order_list = OrderList.objects.filter(sch_date__gte=date.strftime("%Y%m%d"),
                                              sch_date__lte=datetime.datetime.today().strftime("%Y%m%d"))

    return sch_date_from.strftime("%Y-%m-%d"), sch_date_to.strftime("%Y-%m-%d"), order_list

def order_list_search(request):
    date = datetime.datetime.today() - timedelta(days=3)
    for i in request.GET:
        request = json.loads(i)
    order_list_result = []

    if request['dateFrom'] != None:
        date_from = request['dateFrom'].replace('-', '')
        date_to = request['dateTo'].replace('-', '')

        order_list = OrderList.objects.filter(sch_date__gte=date_from).filter(sch_date__lte=date_to)
        order_list = OrderList.objects.raw(
            "SELECT order_id, prod_id, SUM(amount) as amount " +
            "FROM order_orderlist " +
            "WHERE sch_date >= '" + date_from + "' AND " +
            "sch_date <= '" + date_to + "'" +
            "GROUP BY prod_id")
        for i in order_list:
            product_dict = {}
            product = Product.objects.get(prod_id=i.prod_id.prod_id)
            product_name = product.prod_name
            product_dict['order_id'] = i.order_id
            product_dict['prod_name'] = product_name
            product_dict['amount'] = int(i.amount)
            order_list_result.append(product_dict)
    else:
        order_list = OrderList.objects.raw(
            "SELECT order_id, prod_id, SUM(amount) as amount " +
            "FROM order_orderlist " +
            "WHERE sch_date >= '" + date.strftime("%Y%m%d") + "' AND " +
            "sch_date <= '" + datetime.datetime.today().strftime("%Y%m%d") + "'" +
            "GROUP BY prod_id")
        for i in order_list:
            product_dict = {}
            product = Product.objects.get(prod_id=i.prod_id.prod_id)
            product_name = product.prod_name
            product_dict['order_id'] = i.order_id
            product_dict['prod_name'] = product_name
            product_dict['amount'] = int(i.amount)
            order_list_result.append(product_dict)

    def json_default(value):
        if isinstance(value, datetime.date):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')

    return HttpResponse(json.dumps(order_list_result, default=json_default, ensure_ascii=False), content_type="application/json")
    # return JsonResponse(list(order_list_result.values()), safe=False)

def fixed_order(request):
    for i in request.GET:
        request = json.loads(i)
    date_from = request['dateFrom']
    date_to = request['dateTo']
    group_name = request['groupName']
    group_id = request['groupId']
    user_name = request['userName']
    user_id = request['userId']

    final_result = []
    name = ''
    for info in User.objects.filter(username=user_name).values():
        name = info['first_name']

    if group_name == 'customer':  # 고객
        result = OrderSchedule.objects.filter(order_id__sch_date__gte=date_from, order_id__sch_date__lte=date_to)
        for q in result:
            if q.use_yn == 'Y':
                if str(q.order_id.cust_id) == user_id:
                    final_dict = {}
                    final_dict['comp_name'] = q.sch_id.comp_id.comp_name  # 회사명
                    final_dict['order_id'] = q.order_id.order_id  # 오더번호
                    final_dict['amount'] = q.order_id.amount  # 오더수량
                    final_dict['cust_name'] = name  # 고객명
                    final_dict['sch_date'] = q.order_id.sch_date  # 주문일자
                    final_dict['exp_date'] = q.order_id.exp_date  # 작업기한
                    final_dict['textile_type'] = q.order_id.textile_type  # 소재유형
                    final_dict['textile_name'] = q.order_id.textile_name  # 소재명
                    final_dict['amount'] = q.order_id.amount  # 주문수량
                    final_result.append(final_dict)

            custlist = []

            for i in range(len(final_result)):
                custlist.append(final_result[i]['cust_name'])
            custset = set(custlist)
            custlist = list(custset)  # 중복제거

            complist = []
            for i in range(len(final_result)):
                complist.append(final_result[i]['comp_name'])
            compset = set(complist)
            complist = list(compset)  # 중복제거
            #

            count = {}
            for i in range(len(final_result)):
                try:
                    count[final_result[i]['cust_name']] += 1
                except:
                    count[final_result[i]['cust_name']] = 1

            last_result = []
            one_cust = []
            for i in count:
                if count[i] > 1:
                    for q in range(len(final_result)):
                        if i == final_result[q]['cust_name']:
                            common_dict = {}
                            common_dict['cust_name'] = final_result[q]['cust_name']  # 고객명
                            common_dict['comp_name'] = final_result[q]['comp_name']  # 회사명
                            common_dict['order_id'] = final_result[q]['order_id']  # 오더번호
                            common_dict['amount'] = final_result[q]['amount']  # 오더수량
                            common_dict['sch_date'] = final_result[q]['sch_date']  # 주문일자
                            common_dict['exp_date'] = final_result[q]['exp_date']  # 작업기한
                            common_dict['textile_type'] = final_result[q]['textile_type']  # 소재유형
                            common_dict['textile_name'] = final_result[q]['textile_name']  # 소재명
                            one_cust.append(common_dict)
                        else:
                            last_result.append(final_result[q])
                    last_result.append(one_cust)
                    last_result.append(complist)

                    # else:
                    # print('null')
                # else:
                last_result = final_result
            # last_result.append(one_cust)
        last_result.append(complist)
    # if len(request.user.groups.values('id')) == 0: # 회사
    if group_name != 'admin' and group_name != 'customer':
        result = OrderSchedule.objects.filter(order_id__sch_date__gte=date_from, order_id__sch_date__lte=date_to,sch_id__comp_id=group_id)
        if len(result) > 0:
            for q in result:
                if q.use_yn == 'Y':
                    for info in User.objects.filter(id=q.order_id.cust_id).values():
                        name = info['first_name']
                    final_dict = {}
                    final_dict['comp_name'] = q.sch_id.comp_id.comp_name  # 회사명
                    final_dict['order_id'] = q.order_id.order_id  # 오더번호
                    final_dict['amount'] = q.order_id.amount  # 오더수량
                    final_dict['cust_name'] = name  # 고객명
                    final_dict['sch_date'] = q.order_id.sch_date  # 주문일자
                    final_dict['exp_date'] = q.order_id.exp_date  # 작업기한
                    final_dict['textile_type'] = q.order_id.textile_type  # 소재유형
                    final_dict['textile_name'] = q.order_id.textile_name  # 소재명
                    final_dict['amount'] = q.order_id.amount  # 주문수량
                    final_result.append(final_dict)
                custlist = []

            for i in range(len(final_result)):
                custlist.append(final_result[i]['cust_name'])
            custset = set(custlist)
            custlist = list(custset)  # 중복제거

            complist = []
            for i in range(len(final_result)):
                complist.append(final_result[i]['comp_name'])
            compset = set(complist)
            complist = list(compset)  # 중복제거

            count = {}
            for i in range(len(final_result)):
                try:
                    count[final_result[i]['cust_name']] += 1
                except:
                    count[final_result[i]['cust_name']] = 1

            last_result = []
            one_cust = []
            if len(custlist) == 1:
                last_result.append(final_result)
                last_result.append(complist)
            else:
                for i in count:
                    if count[i] > 1: # 동일인의 주문 조회시 리스트로 합쳐서 출력
                        for q in range(len(final_result)):
                            if i == final_result[q]['cust_name']:
                                common_dict = {}
                                common_dict['cust_name'] = final_result[q]['cust_name']  # 고객명
                                common_dict['comp_name'] = final_result[q]['comp_name']  # 회사명
                                common_dict['order_id'] = final_result[q]['order_id']  # 오더번호
                                common_dict['amount'] = final_result[q]['amount']  # 오더수량
                                common_dict['sch_date'] = final_result[q]['sch_date']  # 주문일자
                                common_dict['exp_date'] = final_result[q]['exp_date']  # 작업기한
                                common_dict['textile_type'] = final_result[q]['textile_type']  # 소재유형
                                common_dict['textile_name'] = final_result[q]['textile_name']  # 소재명
                                one_cust.append(common_dict)
                            else:
                                last_result.append(final_result[q])
                        last_result.append(one_cust)

                last_result.append(complist)
        else:
            return JsonResponse({"message": 'null'})
    if group_name == 'admin':  # 관리자
        result = OrderSchedule.objects.filter(order_id__sch_date__gte=date_from, order_id__sch_date__lte=date_to)
        for q in result:
            if (q.use_yn == 'Y'):
                for info in User.objects.filter(id=q.order_id.cust_id).values():
                    name = info['first_name']
                final_dict = {}
                final_dict['comp_name'] = q.sch_id.comp_id.comp_name  # 회사명
                final_dict['order_id'] = q.order_id.order_id  # 오더번호
                final_dict['amount'] = q.order_id.amount  # 오더수량
                final_dict['cust_name'] = name  # 고객명
                final_dict['sch_date'] = q.order_id.sch_date  # 주문일자
                final_dict['exp_date'] = q.order_id.exp_date  # 작업기한
                final_dict['textile_type'] = q.order_id.textile_type  # 소재유형
                final_dict['textile_name'] = q.order_id.textile_name  # 소재명
                final_dict['amount'] = q.order_id.amount  # 주문수량
                final_result.append(final_dict)
        custlist = []

        for i in range(len(final_result)):
            custlist.append(final_result[i]['cust_name'])
        custset = set(custlist)
        custlist = list(custset)  # 중복제거

        complist = []
        for i in range(len(final_result)):
            complist.append(final_result[i]['comp_name'])
        compset = set(complist)
        complist = list(compset)  # 중복제거
        #

        count = {}
        for i in range(len(final_result)):
            try:
                count[final_result[i]['cust_name']] += 1
            except:
                count[final_result[i]['cust_name']] = 1

        last_result = []
        one_cust = []
        for i in count:
            if count[i] > 1:
                for q in range(len(final_result)):
                    if i == final_result[q]['cust_name']:
                        common_dict = {}
                        common_dict['cust_name'] = final_result[q]['cust_name']  # 고객명
                        common_dict['comp_name'] = final_result[q]['comp_name']  # 회사명
                        common_dict['order_id'] = final_result[q]['order_id']  # 오더번호
                        common_dict['amount'] = final_result[q]['amount']  # 오더수량
                        common_dict['sch_date'] = final_result[q]['sch_date']  # 주문일자
                        common_dict['exp_date'] = final_result[q]['exp_date']  # 작업기한
                        common_dict['textile_type'] = final_result[q]['textile_type']  # 소재유형
                        common_dict['textile_name'] = final_result[q]['textile_name']  # 소재명
                        one_cust.append(common_dict)
                    else:
                        last_result.append(final_result[q])
                last_result.append(one_cust)

        last_result.append(complist)

    def json_default(value):
        if isinstance(value, datetime.date):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')

    return HttpResponse(json.dumps(last_result, default=json_default, ensure_ascii=False), content_type="application/json")

# csv file upload
def upload_file(request):
    if request.method == 'POST':
        form = FileUploadCsv(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return JsonResponse({"message": 'success'})
    else:
        form = FileUploadCsv()
    return JsonResponse({"message": request.FILES})

# Draw table after reading csv file
def order_read_csv(request):
    readFile = request.FILES['file'];
    read = pd.read_csv('./media/' + readFile.name, encoding='UTF8')
    data_list = []

    # 예외처리
    # 1. 데이터 컬럼 일치하지 않을 때
    for col in read.columns:
        if '고객명' in col:
            col = read[['고객명', '주문일자', '마감기한', '제품명', '수량', '전화번호', '이메일']]
            for row in range(int(col.size / 7)):
                data_dict = {}
                data_dict['cust_name'] = str(col.loc[[row], ['고객명']].values).replace('[', '').replace(']', '').replace("'", '')
                data_dict['sch_date'] = str(col.loc[[row], ['주문일자']].values).replace('[', '').replace(']', '').replace("'", '')
                data_dict['exp_date'] = str(col.loc[[row], ['마감기한']].values).replace('[', '').replace(']', '').replace("'", '')
                data_dict['prod_name'] = str(col.loc[[row], ['제품명']].values).replace('[', '').replace(']', '').replace("'", '')
                data_dict['amount'] = str(col.loc[[row], ['수량']].values).replace('[', '').replace(']', '').replace("'", '')
                data_dict['contact'] = str(col.loc[[row], ['전화번호']].values).replace('[', '').replace(']', '').replace("'", '')
                data_dict['email'] = str(col.loc[[row], ['이메일']].values).replace('[', '').replace(']', '').replace("'", '')
                data_list.append(data_dict)
    def json_default(value):
        if isinstance(value, datetime.date):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')

    return HttpResponse(json.dumps(data_list, default=json_default, ensure_ascii=False), content_type="application/json")

# Draw table after delete the data
def order_delete_read_csv(request):

    if request.method == 'POST':
        request = json.loads(request.body)
        name = request['name']
        amount = request['amount']
        fileName = request['fileName']

    read = pd.read_csv('./media/' + fileName, encoding='UTF8')
    data_list = []

    # 예외처리
    # 1. 데이터 컬럼 일치하지 않을 때
    for col in read.columns:
        if '고객명' in col:
            col = read[['고객명', '주문일자', '마감기한', '제품명', '수량', '전화번호', '이메일']]
            for row in range(int(col.size / 7)):
                if str(col.loc[[row], ['고객명']].values).replace('[', '').replace(']', '').replace("'", '') == name and str(col.loc[[row], ['수량']].values).replace('[', '').replace(']', '').replace("'", '') == amount:
                    continue;
                else:
                    data_dict = {}
                    data_dict['cust_name'] = str(col.loc[[row], ['고객명']].values).replace('[', '').replace(']', '').replace("'", '')
                    data_dict['sch_date'] = str(col.loc[[row], ['주문일자']].values).replace('[', '').replace(']', '').replace("'", '')
                    data_dict['exp_date'] = str(col.loc[[row], ['마감기한']].values).replace('[', '').replace(']', '').replace("'", '')
                    data_dict['prod_name'] = str(col.loc[[row], ['제품명']].values).replace('[', '').replace(']', '').replace("'", '')
                    data_dict['amount'] = str(col.loc[[row], ['수량']].values).replace('[', '').replace(']', '').replace("'", '')
                    data_dict['contact'] = str(col.loc[[row], ['전화번호']].values).replace('[', '').replace(']', '').replace("'", '')
                    data_dict['email'] = str(col.loc[[row], ['이메일']].values).replace('[', '').replace(']', '').replace("'", '')
                    data_list.append(data_dict)
    def json_default(value):
        if isinstance(value, datetime.date):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')

    return HttpResponse(json.dumps(data_list, default=json_default, ensure_ascii=False), content_type="application/json")


# Save data into the Database by reading csv file
def order_update_csv(request):
    if request.method == 'POST':
        for i in range(len(request.POST.getlist('cust_name'))):
            # id generator
            id_count = OrderList.objects.all().order_by('order_id').last()
            if id_count is None:
                int_id = 0
            else:
                int_id = id_count.order_id[3:]
            str_id = id_generate('ORD', int_id)
            print(request.POST.getlist('prod_name')[i])
            prod_name = Product.objects.get(prod_name=request.POST.getlist('prod_name')[i])
            print('prod_name: ', prod_name)
            OrderList.objects.create(
                order_id=str_id,
                cust_name=request.POST.getlist('cust_name')[i],
                sch_date=request.POST.getlist('sch_date')[i],
                exp_date=request.POST.getlist('exp_date')[i],
                prod_id=Product.objects.get(prod_id=prod_name.prod_id),
                amount=request.POST.getlist('amount')[i],
                contact=request.POST.getlist('contact')[i],
                email=request.POST.getlist('email')[i],
            )

    return redirect("/order/list/")


def order_test(request):


    return JsonResponse({"message": 'error'})
