from django.shortcuts import render
from django.http.response import HttpResponse
import datetime, json
from datetime import timedelta

# Create your views here.
def index(request):
    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '주문관리',
        'selected' : 'ordermanagement'
    }
    return render(request, 'main/index.html', date)

# board
def board(request):
    return render(request, 'main/board.html')

# login
# def login(request):
#     return render(request, 'main/login.html')

# logout
def logout(request):
    return render(request, 'common/logout.html')

# nav 메뉴
def nav(request):
    return render(request, 'common/nav.html')

# topbar - 검색 바
def topbar(request):
    return render(request, 'common/topbar.html')

# footer
def footer(request):
    return render(request, 'common/footer.html')

# compregist
def company(request):
    return render(request, 'company/compregist.html')

