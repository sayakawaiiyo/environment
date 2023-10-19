from predict import *
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

def Environment_predict(f1):
    enviroment_result = environment_predict(f1)
    return enviroment_result


@app.route('/req/environment', methods=['POST','GET'])
def environment():
    uploaded_file = request.files['environment']
    if uploaded_file.filename != '':
        matching_lines = ""
        # Open the file and read its content
        with open("./Data/Datasets/SUN397/val.txt", 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            if str(uploaded_file.filename) in line:
                matching_lines += line
        matching_lines = matching_lines[:-1]
        #print(matching_lines)
        #file_path = os.path.join('./Data/Datasets/SUN397/',matching_lines)
        file_path = os.path.join('temp','temp.jpg')
        uploaded_file.save(file_path)
    return '所选图片'+str(uploaded_file.filename)+'中的图片最有可能的标签为'+Environment_predict(matching_lines)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9999, debug=True)