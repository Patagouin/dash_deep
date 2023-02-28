import dash
#import dash_auth

import users
import os

import dash_html_components as html

from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine

import dash_bootstrap_components as dbc

########### MODEL INSTANCIATION FOR ALL APPS #############
import Models.Shares as sm

shM = sm.Shares(readOnlyThosetoUpdate=False)
########################################################


cssFiles = "assets/style.css"

app = dash.Dash(__name__, suppress_callback_exceptions=True)




#CERULEAN = _BOOTSWATCH_BASE + "cerulean/bootstrap.min.css"
#COSMO = _BOOTSWATCH_BASE + "cosmo/bootstrap.min.css"
#CYBORG = _BOOTSWATCH_BASE + "cyborg/bootstrap.min.css"
#DARKLY = _BOOTSWATCH_BASE + "darkly/bootstrap.min.css"
#FLATLY = _BOOTSWATCH_BASE + "flatly/bootstrap.min.css"
#JOURNAL = _BOOTSWATCH_BASE + "journal/bootstrap.min.css"
#LITERA = _BOOTSWATCH_BASE + "litera/bootstrap.min.css"
#LUMEN = _BOOTSWATCH_BASE + "lumen/bootstrap.min.css"
#LUX = _BOOTSWATCH_BASE + "lux/bootstrap.min.css"
#MATERIA = _BOOTSWATCH_BASE + "materia/bootstrap.min.css"
#MINTY = _BOOTSWATCH_BASE + "minty/bootstrap.min.css"
#PULSE = _BOOTSWATCH_BASE + "pulse/bootstrap.min.css"
#SANDSTONE = _BOOTSWATCH_BASE + "sandstone/bootstrap.min.css"
#SIMPLEX = _BOOTSWATCH_BASE + "simplex/bootstrap.min.css"
#SKETCHY = _BOOTSWATCH_BASE + "sketchy/bootstrap.min.css"
#SLATE = _BOOTSWATCH_BASE + "slate/bootstrap.min.css"
#SOLAR = _BOOTSWATCH_BASE + "solar/bootstrap.min.css"
#SPACELAB = _BOOTSWATCH_BASE + "spacelab/bootstrap.min.css"
#SUPERHERO = _BOOTSWATCH_BASE + "superhero/bootstrap.min.css"
#UNITED = _BOOTSWATCH_BASE + "united/bootstrap.min.css"
#YETI = _BOOTSWATCH_BASE + "yeti/bootstrap.min.css"

#auth = dash_auth.BasicAuth(
#    app,
#    users.VALID_USERNAME_PASSWORD_PAIRS
#)


#sqlEngine = create_engine('postgresql+psycopg2://postgres:Rapide23$@localhost/flask_stocksprices', pool=NullPool)


# Only necessary for export
server = app.server