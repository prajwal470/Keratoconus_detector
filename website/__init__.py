from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'D:/Working Directory/PycharmProjects/KeratoconusDetector/Web Site/website/Uploads'
    app.config['SECRET_KEY'] = '12345'

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    return app