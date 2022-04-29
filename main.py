from flask import Flask
from template_api import template_api

app = Flask(__name__)

app.register_blueprint(template_api, url_prefix='/template')

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run()