from flask import Flask, jsonify, request
from detect import detect
import time


app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({'msg': "OK"}), 200


@app.route('/api/detect', methods=['POST'])
def predict():
    media = request.files.get('file')
    mimetype = media.content_type
    # print(mimetype)
    if 'video' in mimetype:
        media_path = f'static/images_up/{str(time.time())}.{media.filename.split(".")[-1]}' 
    elif 'image' in mimetype:    
        media_path = f'static/images_up/{str(time.time())}.{media.filename.split(".")[-1]}'
    else: 
        return jsonify({'msg': 'image or video'})
    media.save(media_path)
    img = detect(media_path)
    return jsonify({"data": img}), 200

if __name__ == "__main__":
    app.run(debug=True)