from flask import Blueprint, render_template, request, url_for, flash, redirect
from PIL import Image

views = Blueprint('views', __name__)

@views.route('/' , methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'datafile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        datafile = request.files['datafile']

        if datafile.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if datafile:
            img = Image.open(datafile)
            img = img.convert('L')  
            img.save('output.png')
            return redirect(url_for('auth.generate_report'))

    
    return render_template("home.html")