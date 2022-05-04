from flask import Flask
from template_api import template_api
from DepthAPI import depth_api
from SegmentationAPI import segmentation_api

app = Flask(__name__)

app.register_blueprint(template_api, url_prefix='/template')
app.register_blueprint(depth_api, url_prefix='/depth')
app.register_blueprint(segmentation_api, url_prefix='/segmentation')

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


