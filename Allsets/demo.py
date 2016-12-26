from flask import Flask, request, render_template
app = Flask(__name__)
#import Image_Rotater
import subprocess
import imghdr


with open('SESSION_SCORE', 'w'): pass #Clear the session score
#demo = Image_Rotater.Image_Rotater()

@app.route("/")
def hello():
    return render_template('demo.html')
 
@app.route("/demo", methods=['POST'])
def echo(): 
    
    url=request.form['url']
    rotation=request.form['rotation']
    lifetime_score = 0
    session_score = 0

    #Do the demo
    if url and rotation:
        #demo.full_demo(url, int(rotation))
	subprocess.call(["python", "demo_test.py", url, rotation])
	filename = url.split('/')[-1].split('#')[0].split('?')[0] # get filename and remove queries
	download_path = "static/" + filename 
	if not (filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
		filename += "." + imghdr.what(download_path)
	rotated_path = "static/rotated_" + filename
	derotated_path = "static/derotated_" + filename

    #Get the stats/scores
    lines = [line.rstrip('\n') for line in open('LIFETIME_SCORE')]
    if lines:
	    lifetime_ones = lines.count("1")
	    lifetime_score = (lifetime_ones*100)/(len(lines))

    lines = [line.rstrip('\n') for line in open('SESSION_SCORE')]
    if lines:
	    session_ones = lines.count("1")
	    session_score = (session_ones*100)/(len(lines))


    return render_template('demo.html', url=url, 
				rotation=rotation, 
				rotated=rotated_path, 
				derotated=derotated_path,
				lifetime_score=lifetime_score,
				session_score=session_score)
 
 
if __name__ == "__main__":
    app.run(debug=True, port=6006, host='0.0.0.0')
