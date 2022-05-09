from flask import Flask
from template_api import template_api
from DepthAPI import depth_api
from SegmentationAPI import segmentation_api
from ClassificationAPI import classification_api
from volume_estimationAPI import volume_estimation_api
from coin_detectionAPI import coin_detection_api

app = Flask(__name__)

app.register_blueprint(template_api, url_prefix="/template")
app.register_blueprint(depth_api, url_prefix="/depth")
app.register_blueprint(segmentation_api, url_prefix="/segmentation")
app.register_blueprint(volume_estimation_api, url_prefix="/volumeEstimation")
app.register_blueprint(classification_api, url_prefix="/classification")
app.register_blueprint(coin_detection_api, url_prefix="/coinDetection")

app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
