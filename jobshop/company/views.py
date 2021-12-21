#-*- coding: utf-8 -*-
from django.urls import reverse_lazy
from django.http import HttpResponse
from django.views.generic import ListView, CreateView, DetailView, UpdateView, DeleteView, TemplateView
from django.http import JsonResponse
from django.shortcuts import render
from django.shortcuts import get_object_or_404
from openpyxl import load_workbook
from fileutils.forms import FileUploadCsv
from .models import *
import pandas as pd
import json, csv, datetime
from datetime import timedelta


def home(request):
    template_name = 'company/compregist.html'
    comp_list = Information.objects.all()
    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        "comp_list": comp_list,
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '업체정보 / 설비정보등록'
    }
    return render(request, template_name, date)
    # model = Company
    # paginate_by = 5
    # ordering = ['id']

def delete(request):
    if request.method == 'POST':
        request = json.loads(request.body)
        id = request['id']
        company = get_object_or_404(Information, pk=id)
        company.delete()
        return HttpResponse(status=200)


def comp_facility_view(request):
    template_name = 'company/compfaclist.html'
    comp_list = Information.objects.all()
    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        "comp_list": comp_list,
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '업체정보 / 설비정보검색'
    }

    return render(request, template_name, date)


def comp_daily_view(request):
    template_name = 'company/compdailyproduction.html'
    comp_list = Information.objects.all()
    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        "comp_list": comp_list,
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '업체정보 / 실제가동현황'
    }
    
    return render(request, template_name, date)

def comp_list_view(request):
    template_name = 'company/compinfolist.html'
    date = datetime.datetime.today() - timedelta(days=3)
    comp_list = Information.objects.all()
    result_list = []
    for i in range(len(comp_list.values())):
        result = {}
        if len(comp_list[i].textile_type.split(',')) > 1:
            k_textile = ''
            w_textile = ''
            textile = ''
            if comp_list[i].textile_type.split(',')[0] == 'weave':
                w_textile = '제직'
            if comp_list[i].textile_type.split(',')[1] == 'weave':
                w_textile = '제직'
            if comp_list[i].textile_type.split(',')[0] == 'knit':
                k_textile = '편직'
            if comp_list[i].textile_type.split(',')[1] == 'knit':
                k_textile = '편직'
            textile = str(w_textile) + ', ' + str(k_textile)
        else:
            if comp_list[i].textile_type.split(',')[0] == 'weave':
                textile = '제직'
            else:
                textile = '편직'
        result['comp_id'] = comp_list[i].comp_id
        result['comp_name'] = comp_list[i].comp_name
        result['facility_count'] = comp_list[i].facility_count
        result['textile_type'] = textile
        result_list.append(result)

    date = {
        "comp_list": result_list,
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '업체정보 / 업체정보검색'
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
    uploadFile = request.FILES['file'];
    # read = pd.read_csv('./media/' + uploadFile.name, encoding='euc-kr')
    read = pd.read_csv('./media/' + uploadFile.name, encoding='cp949')

    # 예외처리
    # 1. 데이터 컬럼 일치하지 않을 때
    for col in read.columns: # 작업일자        회사명  작업 호기       제품명  RPM     가동시간(HHMM)  운행시간(HHMM)  생산량(yd)      가동율  작업종료일자
        if '작업일자' in col and '회사명' in col and '작업 호기' in col:
            col = read[['작업일자', '회사명', '작업 호기', '제품명', 'RPM', '가동시간(HHMM)', '운행시간(HHMM)', '생산량(yd)', '가동율', '작업종료일자']]
            for row in range(int(col.size / 10)):
                Information.objects.create(
                    comp_name=str(col.loc[[row], ['회사명']].values).replace('[', '').replace(']', '').replace("'", ''),
                    work_date=str(col.loc[[row], ['작업일자']].values).replace('[', '').replace(']', '').replace("'", ''),
                    work_end_date=str(col.loc[[row], ['작업종료일자']].values).replace('[', '').replace(']', '').replace("'", ''),
                    fac_code=str(col.loc[[row], ['작업 호기']].values).replace('[', '').replace(']', '').replace("'", ''),
                    prod_name=str(col.loc[[row], ['제품명']].values).replace('[', '').replace(']', '').replace("'", ''),
                    rpm=str(col.loc[[row], ['RPM']].values).replace('[', '').replace(']', '').replace("'", ''),
                    uptime=str(col.loc[[row], ['가동시간(HHMM)']].values).replace('[', '').replace(']', '').replace("'", ''),
                    running_time=str(col.loc[[row], ['운행시간(HHMM)']].values).replace('[', '').replace(']', '').replace("'", ''),
                    prod_output=str(col.loc[[row], ['생산량(yd)']].values).replace('[', '').replace(']', '').replace("'",  ''),
                    prod_rate=str(col.loc[[row], ['가동율']].values).replace('[', '').replace(']', '').replace("'", '')
                )
                print('CATEGORY DATA UPLOADED SUCCESSFULY!')
            return JsonResponse({"message": 'success'})
        else:
            return JsonResponse({"message": 'error'})




