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
        has_pills = [
            "Unknown" if current_slots_data[i]['lid'] == "Close" else
            ("Full" if current_slots_data[i]['Has_pill'] else "Empty")
            for i in range(4)
        ]
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
        
def save_frame(frame, current_slots_data, tracker, duration, missing):
    current_opens = [
        idx for idx, 
        data in current_slots_data.items() 
        if data['lid'] == "Open"
    ]

    if len(current_opens) > 0:
        tracker['missing_start_time'] = None

        if current_opens != tracker['active_opens']:
            tracker['active_opens'] = current_opens
            tracker['open_start_time'] = t.time()
            tracker['triggered'] = False

        else:
            if tracker['open_start_time'] is not None and not tracker['triggered']:
                elapsed_time = t.time() - tracker['open_start_time']

                if elapsed_time > duration:
                    slot_details = []#will delete at fininal
                    for idx in current_opens:#will delete at fininal
                        pill_state = "Full" if current_slots_data[idx]['Has_pill'] else "Empty"#will delete at fininal
                        slot_details.append(f"slot{idx}_{pill_state}")#will delete at fininal
                    slots_str = "_".join(slot_details)#will delete at fininal
                    timestamp = t.strftime("%Y%m%d_%H%M%S")#will delete at fininal
                    filename = f"saved_slots/{timestamp}_{slots_str}.png"#will delete at fininal
                    cv2.imwrite(filename, frame)#will delete at fininal

                    db_manager.insert_pill_data(current_slots_data, frame)
                    tracker['triggered'] = True
    
    else:
        if tracker['open_start_time'] is not None:
            if tracker['missing_start_time'] is None:
                tracker['missing_start_time'] = t.time()

            lost_duration = t.time() - tracker['missing_start_time']

            if lost_duration > missing:
                tracker['active_opens'] = []
                tracker['open_start_time'] = None
                tracker['missing_start_time'] = None
                tracker['triggered'] = False

db_manager = PillManager()
