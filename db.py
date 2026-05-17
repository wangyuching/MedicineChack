import cv2
import time as t
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Pill(db.Model):
    __tablename__ = 'pills'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    dt = db.Column(db.DateTime, nullable=False)

    lid0 = db.Column(db.String(10), default="Unknown")
    lid1 = db.Column(db.String(10), default="Unknown")
    lid2 = db.Column(db.String(10), default="Unknown")
    lid3 = db.Column(db.String(10), default="Unknown")

    has_pill0 = db.Column(db.String(10), default="Unknown")
    has_pill1 = db.Column(db.String(10), default="Unknown")
    has_pill2 = db.Column(db.String(10), default="Unknown")
    has_pill3 = db.Column(db.String(10), default="Unknown")

    img = db.Column(db.LargeBinary(length=(2**24)-1), nullable=False)

class PillManager:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:DataBase@127.0.0.1:3306/project"
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

        db.init_app(self.app)

        with self.app.app_context():
            db.create_all()
    
    def insert_pill_data(self, current_slots_data, frame):
        dt_str = t.strftime("%Y-%m-%d %H:%M:%S", t.localtime())

        lids = [current_slots_data[i]['lid'] for i in range(4)]
        has_pills = [current_slots_data[i]['Has_pill'] for i in range(4)]
        # has_pills = ["Full" if current_slots_data[i]['Has_pill'] else "Empty" for i in range(4)]
        _, buffer = cv2.imencode('.jpg', frame)
        img_data = buffer.tobytes()

        with self.app.app_context():
            try:
                new_data = Pill(
                    dt=dt_str,
                    lid0=lids[0],
                    lid1=lids[1],
                    lid2=lids[2],
                    lid3=lids[3],
                    has_pill0=has_pills[0],
                    has_pill1=has_pills[1],
                    has_pill2=has_pills[2],
                    has_pill3=has_pills[3],
                    img=img_data
                )
                db.session.add(new_data)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                print(f"Error inserting data: {e}")
        
