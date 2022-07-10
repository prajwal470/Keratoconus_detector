from flask import Blueprint, render_template, redirect, url_for, request, Flask
import Classifier
import PreProcessing

auth = Blueprint('auth', __name__)


@auth.route('/generate', methods=['GET', 'POST'])
def generate_report():
    if request.method == 'POST':
        return redirect(url_for('auth.report'))
    return render_template("generate.html")


@auth.route('/report', methods=['GET', 'POST'])
def report():
    status = (Classifier.classify('output.png'))
    angle, thickness = PreProcessing.features('output.png')
    return render_template("report.html", STATUS = status, Curvature_angle = angle, Cornea_thickness = thickness)
