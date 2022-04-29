from flask import Blueprint

template_api = Blueprint('template_api', __name__)

@template_api.route("/")
def accountList():
    return "test"