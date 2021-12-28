#-*- coding: utf-8 -*-
from django.urls import reverse_lazy
from django.http import HttpResponse
from django.views.generic import ListView, CreateView, DetailView, UpdateView, DeleteView, TemplateView
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.utils.encoding import smart_str
from openpyxl import load_workbook
from fileutils.forms import FileUploadCsv
from fileutils.models import FileUploadCsv as fileUploadCsv
from .models import *
from common.views import id_generate
import pandas as pd
import json, csv, datetime
from urllib.parse import quote
from datetime import timedelta

# 설비정보등록 company Facility register HTML
def home(request):
    template_name = 'company/comp_regist.html'
    comp_list = Facility.objects.all()
    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        "comp_list": comp_list,
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '업체정보 / 설비정보등록'
    }
    return render(request, template_name, date)

# 설비정보검색 company Facility view HTML
def comp_list_view(request):
    template_name = 'company/comp_fac_list.html'
    date = datetime.datetime.today() - timedelta(days=3)
    comp_list = Facility.objects.filter(comp_id=request.user.groups.values('id')[0]['id'])

    result_list = []
    for i in range(len(comp_list.values())):
        # comp_name = Information.objects.get(comp_id=comp_list[i].comp_id)
        comp_name = comp_list[i].comp_id.comp_name # comp_name.name
        result = {}
        result['comp_name'] = comp_name
        result['facility_name'] = comp_id = comp_list[i].facility_name
        result_list.append(result)

    date = {
        "comp_list": result_list,
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '업체정보 / 설비정보검색'
    }
    return render(request, template_name, date)

# 설비현황검색 company production view HTML
def comp_facility_view(request):
    template_name = 'company/comp_prod_list.html'
    comp_list = Facility.objects.all()
    result_list = []
    for i in range(len(comp_list.values())):
        comp_name = comp_list[i].comp_id.comp_name  # comp_name.name
        result = {}
        result['comp_name'] = comp_name
        result['facility_name'] = comp_id = comp_list[i].facility_name
        result_list.append(result)

    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        "comp_list": result_list,
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '업체정보 / 설비현황검색'
    }

    return render(request, template_name, date)

def delete(request):
    if request.method == 'POST':
        request = json.loads(request.body)
        id = request['id']
        company = get_object_or_404(Facility, pk=id)
        company.delete()
        return HttpResponse(status=200)

# 제품정보등록 company product register HTML
def comp_product_regist(request):
    template_name = 'company/comp_product_regist.html'
    comp_list = Facility.objects.all()
    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        "comp_list": comp_list,
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '업체정보 / 제품정보등록'
    }

    return render(request, template_name, date)

# 제품정보검색 company product view HTML
def comp_product_view(request):
    template_name = 'company/comp_product_list.html'
    comp_list = Product.objects.filter(comp_id=request.user.groups.values('id')[0]['id'])
    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        "comp_list": comp_list,
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '업체정보 / 제품정보검색'
    }
    
    return render(request, template_name, date)

# csv 파일 업로드
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
def prod_read_csv(request):
    readFile = fileUploadCsv.objects.all().order_by('id').last()

    # request.FILES['file'];
    read = pd.read_csv('./media/' + str(readFile.file), encoding='UTF8')
    data_list = []

    # 예외처리
    # 1. 데이터 컬럼 일치하지 않을 때
    for col in read.columns:
        if '제품명' in col:
            col = read[['제품명', 'rpm', '밀도', '일일생산량']]
            for row in range(int(col.size / 4)):
                data_dict = {}
                data_dict['prod_name'] = str(col.loc[[row], ['제품명']].values).replace('[', '').replace(']', '').replace("'", '')
                data_dict['density'] = str(col.loc[[row], ['밀도']].values).replace('[', '').replace(']', '').replace("'", '')
                data_dict['rpm'] = str(col.loc[[row], ['rpm']].values).replace('[', '').replace(']', '').replace("'", '')
                data_dict['daily_prod_rate'] = str(col.loc[[row], ['일일생산량']].values).replace('[', '').replace(']', '').replace("'", '')
                data_list.append(data_dict)
    def json_default(value):
        if isinstance(value, datetime.date):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')

    return HttpResponse(json.dumps(data_list, default=json_default, ensure_ascii=False), content_type="application/json")

def prod_csv_download(request):
    filename = request.user.groups.values('name')[0]['name'] + "_제품정보.csv"

    # response content type
    response = HttpResponse(
        content_type='text/csv',
        headers={'Content-Disposition': "attachment;filename*=UTF-8''{}".format(quote(filename.encode('utf-8')))},
    )

    writer = csv.writer(response, csv.excel)
    response.write(u'\ufeff'.encode('utf8'))

    # write the headers
    writer.writerow([
        smart_str(u"제품명"),
        smart_str(u"rpm"),
        smart_str(u"밀도"),
        smart_str(u"일일생산량"),
    ])
    # get data from database or from text file....
    products = Product.objects.filter(comp_id=request.user.groups.values('id')[0]['id'])
    if not products:
        return response
    else:
        for product in products:
            writer.writerow([
                smart_str(product.prod_name),
                smart_str(product.rpm),
                smart_str(product.density),
                smart_str(product.daily_prod_rate),
            ])

    return response

# 제품문서양식다운로드
def prod_csv_download_blank(request):
    filename = request.user.groups.values('name')[0]['name'] + "_제품정보.csv"

    # response content type
    response = HttpResponse(
        content_type='text/csv',
        headers={'Content-Disposition': "attachment;filename*=UTF-8''{}".format(quote(filename.encode('utf-8')))},
    )

    writer = csv.writer(response, csv.excel)
    response.write(u'\ufeff'.encode('utf8'))

    # write the headers
    writer.writerow([
        smart_str(u"제품명"),
        smart_str(u"rpm"),
        smart_str(u"밀도"),
        smart_str(u"일일생산량"),
    ])
    return response

# Save data into the Database by reading csv file
def comp_prod_update_csv(request):
    if request.method == 'POST':
        for i in range(len(request.POST.getlist('prod_name'))):
            # id generator
            id_count = Product.objects.all().order_by('prod_id').last()
            if id_count is None or not id_count:
                int_id = 0
            else:
                int_id = id_count.prod_id[3:]
            str_id = id_generate('PRD', int_id)
            Product.objects.create(
                prod_id=str_id,
                comp_id=Information.objects.get(comp_id=request.user.groups.values('id')[0]['id']),
                prod_name=request.POST.getlist('prod_name')[i],
                density=request.POST.getlist('density')[i],
                rpm=request.POST.getlist('rpm')[i],
                daily_prod_rate=request.POST.getlist('daily_prod_rate')[i]
            )

    return redirect("/company/product/search/")

# Draw table after reading csv file
def comp_read_csv(request):
    readFile = fileUploadCsv.objects.all().order_by('id').last()

    # readFile = request.FILES['file'];
    read = pd.read_csv('./media/' + str(readFile.file), encoding='UTF8')
    data_list = []

    # 예외처리
    # 1. 데이터 컬럼 일치하지 않을 때
    for col in read.columns:
        if '호기명' in col:
            col = read[['호기명']]
            for row in range(int(col.size)):
                data_dict = {}
                data_dict['facility_name'] = str(col.loc[[row], ['호기명']].values).replace('[', '').replace(']', '').replace("'", '')
                data_list.append(data_dict)
    def json_default(value):
        if isinstance(value, datetime.date):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')

    return HttpResponse(json.dumps(data_list, default=json_default, ensure_ascii=False), content_type="application/json")



# 설비문서양식다운로드
def comp_csv_download_blank(request):
    filename = request.user.groups.values('name')[0]['name'] + "_설비정보.csv"

    # response content type
    response = HttpResponse(
        content_type='text/csv',
        headers={'Content-Disposition': "attachment;filename*=UTF-8''{}".format(quote(filename.encode('utf-8')))},
    )

    writer = csv.writer(response, csv.excel)
    response.write(u'\ufeff'.encode('utf8'))

    # write the headers
    writer.writerow([
        smart_str(u"호기명"),
    ])
    return response

def comp_csv_download(request):
    filename = request.user.groups.values('name')[0]['name'] + "_제품정보.csv"

    # response content type
    response = HttpResponse(
        content_type='text/csv',
        headers={'Content-Disposition': "attachment;filename*=UTF-8''{}".format(quote(filename.encode('utf-8')))},
    )

    writer = csv.writer(response, csv.excel)
    response.write(u'\ufeff'.encode('utf8'))

    # write the headers
    writer.writerow([
        smart_str(u"제품명"),
        smart_str(u"rpm"),
        smart_str(u"밀도"),
        smart_str(u"일일생산량"),
    ])
    # get data from database or from text file....
    products = Product.objects.filter(comp_id=request.user.groups.values('id')[0]['id'])
    if not products:
        return response
    else:
        for product in products:
            writer.writerow([
                smart_str(product.prod_name),
                smart_str(product.rpm),
                smart_str(product.density),
                smart_str(product.daily_prod_rate),
            ])

    return response


# excel 파일 업로드
def comp_update_excel(request):
    if request.method == 'POST':

        file = request.FILES['file']
        # data_only=Ture로 해줘야 수식이 아닌 값으로 받아온다.
        load_wb = load_workbook(file, data_only=True)

        # 시트 이름으로 불러오기
        load_ws = load_wb['Sheet1']

        # 셀 주소로 값 출력
        # print(load_ws['A1'].value)

        # 일단 리스트에 담기
        all_values = []
        for row in load_ws.rows:
            row_value = []
            for cell in row:
                row_value.append(cell.value)
            all_values.append(row_value)
            print(all_values)

        cnt = 0
        for idx, val in enumerate(all_values):
            if idx == 0:
                # 엑셀 형식 체크 (첫번째의 제목 row)
                if val[0] != '항목1' or val[1] != '항목2' or val[2] != '항목3':
                    context = {'state': False, 'rtnmsg': '엑셀 항목이 올바르지 않습니다.'}
                    return HttpResponse(json.dumps(context), content_type="application/json")
            else:
                # print(type(val[2]))
                if val[2] and type(val[2]) == int:
                    memData = Member.objects.get(msabun=val[0], mname=val[1])
                    memData.myear_salary = val[2]
                    memData.save()
                    cnt += 1

        context = {'state': True, 'rtnmsg': '{0}건의 엑셀 데이터가 반영 되었습니다.'.format(cnt)}
        return HttpResponse(json.dumps(context), content_type="application/json")

# csv 파일 읽어서 데이터베이스에 저장
def comp_update_csv(request):        # 파일객체가 하나도 없다면 작업을 멈추고 리턴합니다.
    if request.method == 'POST':
        for i in range(len(request.POST.getlist('facility_name'))):
            # id generator
            str_id = 'FAC' + str(request.user.groups.values('id')[0]['id']) + '#' + request.POST.getlist('facility_name')[i].replace("호기", "")
            Facility.objects.create(
                facility_id=str_id,
                facility_name=request.POST.getlist('facility_name')[i].replace("호기", ""),
                comp_id=Information.objects.get(comp_id=request.user.groups.values('id')[0]['id'])
            )

    return redirect("/company/info/list/")

# 해당 일자 작업 가능한 기계 조회
def comp_avail_facility(request):

    for i in request.GET:
        request = json.loads(i)

    if request['strDate'] != None:
        strDate = request['strDate'].replace('-', '')
        order_list = Schedule.objects.filter(work_end_date__gt=strDate)
    else:
        order_list = Schedule.objects.filter(work_end_date__gt=datetime.datetime.today().strftime("%Y%m%d"))

    fac_list = Facility.objects.all()
    result_list = []
    duplicate = []
    if len(order_list) > 0:
        for i in fac_list:
            for j in order_list:
                if i.facility_id == j.facility_id.facility_id:
                    continue;
                else:
                    result_list.append(i.facility_id)
        result = []
        final_list = []
        for i in result_list:
            if i in result:
                final_list.append(i)
            if i not in result:
                result.append(i)
    else:
        final_list = []
        for i in fac_list:
            final_list.append(i.facility_id)



    return JsonResponse(final_list, safe=False)
