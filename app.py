from flask import Flask, jsonify
from detect import detect
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'msg': "OK"}), 200

