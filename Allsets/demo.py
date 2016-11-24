from flask import Flask, request, render_template
app = Flask(__name__)
#import Image_Rotater
import subprocess
 
#demo = Image_Rotater.Image_Rotater()

@app.route("/")
def hello():
    return render_template('demo.html')
 
@app.route("/demo", methods=['POST'])
def echo(): 
    
    url=request.form['url']
    rotation=request.form['rotation']
    if url and rotation:
        #demo.full_demo(url, int(rotation))
	subprocess.call(["python", "demo_test.py", url, rotation])
	filename = url.split('/')[-1].split('#')[0].split('?')[0] # get filename and remove queries
	rotated_path = "static/rotated_" + filename 
	derotated_path = "static/derotated_" + filename 
    return render_template('demo.html', url=url, rotation=rotation, rotated=rotated_path, derotated=derotated_path)
 
 
if __name__ == "__main__":
    app.run(debug=True, port=6006, host='0.0.0.0')
