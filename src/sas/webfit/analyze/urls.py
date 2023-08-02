import logging
from django.urls import path, re_path, include
from . import views

logger = logging.getLogger(__name__)

urlpatterns = [
    #path("<str:username>/<int:data_id>/", views.list_analysis_done, "view analysis done"),
    
    path("fit/", include("analyze.fitting.urls"), name = "fit patterns"),
    #path("Inversion/", include("analyze.inversion.urls")),   <- where is this in the main script? 
    #path("Invariant/", include("analyze.invariant.urls")),
    #path("Corfunc/", include("analyze.corfunc.urls")),
]