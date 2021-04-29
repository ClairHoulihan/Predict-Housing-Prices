from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home(request):
    return render(request, 'predictions/home.html')

def graph(request):
    return render(request, 'predictions/graph.html', {'title': 'Graph'})