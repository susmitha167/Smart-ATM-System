from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
from datetime import datetime
from datetime import date
import datetime
import random
from random import seed
from random import randint
import cv2
import numpy as np
import threading
import os
import time
import shutil
import imagehash
import PIL.Image
from PIL import Image
from PIL import ImageTk
import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="smart_atm"
)


app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
@app.route('/',methods=['POST','GET'])
def index():
    cnt=0
    act=""
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    ff11=open("img.txt","w")
    ff11.write("1")
    ff11.close()
        

    return render_template('index.html',msg=msg,act=act)

@app.route('/verify_card',methods=['POST','GET'])
def verify_card():
    cnt=0
    act=""
    msg=""
    
    if request.method=='POST':
        card=request.form['card']
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM register where card=%s",(card, ))
        cnt = mycursor.fetchone()[0]
        if cnt>0:
            msg="success"
            session['username'] = card
            ff2=open("un.txt","w")
            ff2.write(card)
            ff2.close()
            return redirect(url_for('verify_face'))
       
            
        else:
            msg="Card No. is wrong!"
            print("Incorrect")
        

    return render_template('verify_card.html',msg=msg,act=act)

#########################

@app.route('/register',methods=['POST','GET'])
def register():
    result=""
    act=""
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        bank=request.form['bank']
        branch=request.form['branch']
        card=request.form['card']
        account=request.form['accno']
        uname=request.form['username']
        password=request.form['password']

        aadhar1=request.form['aadhar1']
        aadhar2=request.form['aadhar2']
        aadhar3=request.form['aadhar3']

        face_st=request.form['face_st']
        
        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM register where card=%s",(card, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO register(id, name, mobile, email, address,  bank, accno, branch, card, deposit, username, password, rdate, aadhar1, aadhar2, aadhar3, face_st, fimg) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, mobile, email, address, bank, account, branch, card, '10000', uname, password, rdate, aadhar1, aadhar2, aadhar3, face_st, '')
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
            if face_st=="1":
                return redirect(url_for('add_photo',vid=maxid))
            #if mycursor.rowcount==1:
            #    result="Registered Success"
            else:
                return redirect(url_for('index',act='success'))
        else:
            result="Card No. already Exist!"
    return render_template('register.html',result=result)

@app.route('/login_admin', methods=['POST','GET'])
def login_admin():
    result=""
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            result=" Your Logged in sucessfully**"
            return redirect(url_for('admin')) 
        else:
            result="Your logged in fail!!!"
                
    
    return render_template('login_admin.html',result=result)

@app.route('/admin',methods=['POST','GET'])
def admin():
    msg=""
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()

    mycursor = mydb.cursor()
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        branch=request.form['branch']
        aadhar=request.form['aadhar']

        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        
        
        mycursor.execute("SELECT count(*) FROM register where aadhar1=%s",(aadhar, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            str1=str(maxid)
            ac=str1.rjust(4, "0")
            account="223344"+ac

            xn=randint(1000, 9999)
            rv1=str(xn)
            xn2=randint(1000, 9999)
            rv2=str(xn2)
            card=rv1+ac+rv2
            bank="SBI"

            xn3=randint(1000, 9999)
            pinno=str(xn3)
            
            
            sql = "INSERT INTO register(id, name, mobile, email, address,  bank, accno, branch, card, deposit,password, rdate, aadhar1) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, mobile, email, address, bank, account, branch, card, '10000',pinno, rdate, aadhar)
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()
            message="Dear "+name+", Your Bank Account created, Account No.:"+account+", Debit Card No."+card+", Pinno:"+pinno
            url="http://iotcloud.co.in/testmail/sendmail.php?email="+email+"&message="+message
            webbrowser.open_new(url)
            
            return redirect(url_for('add_photo',vid=maxid)) 
        else:
            msg="Already Exist!"

    mycursor.execute("SELECT amount FROM admin WHERE username='admin'")
    value = mycursor.fetchone()[0]
    
    return render_template('admin.html',msg=msg,value=value)

@app.route('/add_photo',methods=['POST','GET'])
def add_photo():
    vid=""
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()
    if request.method=='GET':
        vid = request.args.get('vid')
        ff=open("user.txt","w")
        ff.write(vid)
        ff.close()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        cursor = mydb.cursor()

        cursor.execute('delete from vt_face WHERE vid = %s', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1=vid+"_"+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM vt_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface=vid+"_"+str(i)+".jpg"
            sql = "INSERT INTO vt_face(id, vid, vface) VALUES (%s, %s, %s)"
            val = (maxid, vid, vface)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        cursor.execute('update register set fimg=%s WHERE id = %s', (vface1, vid))
        mydb.commit()
        shutil.copy('faces/f1.jpg', 'static/photo/'+vface1)
        return redirect(url_for('view_cus',vid=vid,act='success'))
        
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM register")
    data = cursor.fetchall()
    return render_template('add_photo.html',data=data, vid=vid)

@app.route('/view_cus',methods=['POST','GET'])
def view_cus():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register")
    value = mycursor.fetchall()
    return render_template('view_cus.html', result=value)

###Preprocessing
@app.route('/view_photo',methods=['POST','GET'])
def view_photo():
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()

    if request.method=='POST':
        print("Training")
        vid=request.form['vid']
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        dt = cursor.fetchall()
        for rs in dt:
            ##Preprocess
            path="static/frame/"+rs[2]
            path2="static/process1/"+rs[2]
            mm2 = PIL.Image.open(path).convert('L')
            rz = mm2.resize((200,200), PIL.Image.ANTIALIAS)
            rz.save(path2)
            
            '''img = cv2.imread(path2) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            path3="static/process2/"+rs[2]
            cv2.imwrite(path3, dst)'''
            ######
            img = cv2.imread(path2)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/"+rs[2]
            segment.save(path3)
            
            #####
            image = cv2.imread(path2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 50, 100)
            image = Image.fromarray(image)
            edged = Image.fromarray(edged)
            path4="static/process3/"+rs[2]
            edged.save(path4)
            ##
            shutil.copy('static/images/11.png', 'static/process4/'+rs[2])
       
        return redirect(url_for('view_photo1',vid=vid))
        
    return render_template('view_photo.html', result=value,vid=vid)

###Segmentation using RNN
def crfrnn_segmenter(model_def_file, model_file, gpu_device, inputs):
    
    assert os.path.isfile(model_def_file), "File {} is missing".format(model_def_file)
    assert os.path.isfile(model_file), ("File {} is missing. Please download it using "
                                        "./download_trained_model.sh").format(model_file)

    if gpu_device >= 0:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_file, model_file, caffe.TEST)

    num_images = len(inputs)
    num_channels = inputs[0].shape[2]
    assert num_channels == 3, "Unexpected channel count. A 3-channel RGB image is exptected."
    
    caffe_in = np.zeros((num_images, num_channels, _MAX_DIM, _MAX_DIM), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        caffe_in[ix] = in_.transpose((2, 0, 1))

    start_time = time.time()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    end_time = time.time()

    print("Time taken to run the network: {:.4f} seconds".format(end_time - start_time))
    predictions = out[net.outputs[0]]

    return predictions[0].argmax(axis=0).astype(np.uint8)


def run_crfrnn(input_file, output_file, gpu_device):
    """ Runs the CRF-RNN segmentation on the given RGB image and saves the segmentation mask.
    Args:
        input_file: Input RGB image file (e.g. in JPEG format)
        output_file: Path to save the resulting segmentation in PNG format
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
    """

    input_image = 255 * caffe.io.load_image(input_file)
    input_image = resize_image(input_image)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    palette = get_palette(256)
    #PIL reads image in the form of RGB, while cv2 reads image in the form of BGR, mean_vec = [R,G,B] 
    mean_vec = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    mean_vec = mean_vec.reshape(1, 1, 3)

    # Rearrange channels to form BGR
    im = image[:, :, ::-1]
    # Subtract mean
    im = im - mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = _MAX_DIM - cur_h
    pad_w = _MAX_DIM - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Get predictions
    segmentation = crfrnn_segmenter(_MODEL_DEF_FILE, _MODEL_FILE, gpu_device, [im])
    segmentation = segmentation[0:cur_h, 0:cur_w]

    output_im = PILImage.fromarray(segmentation)
    output_im.putpalette(palette)
    output_im.save(output_file)
###Feature extraction & Classification
def DCNN_process(self):
        
        train_data_preprocess = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

        test_data_preprocess = (1./255)

        train = train_data_preprocess.flow_from_directory(
                'dataset/training',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        test = train_data_preprocess.flow_from_directory(
                'dataset/test',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        ## Initialize the Convolutional Neural Net

        # Initialising the CNN
        cnn = Sequential()

        # Step 1 - Convolution
        # Step 2 - Pooling
        cnn.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second convolutional layer
        cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        cnn.add(Flatten())

        # Step 4 - Full connection
        cnn.add(Dense(units = 128, activation = 'relu'))
        cnn.add(Dense(units = 1, activation = 'sigmoid'))

        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        history = cnn.fit_generator(train,
                                 steps_per_epoch = 250,
                                 epochs = 25,
                                 validation_data = test,
                                 validation_steps = 2000)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        test_image = image.load_img('\\dataset\\', target_size=(128,128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        print(result)

        if result[0][0] == 1:
                print('feature extracted and classified')
        else:
                print('none')
                
@app.route('/view_photo1',methods=['POST','GET'])
def view_photo1():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo1.html', result=value,vid=vid)

@app.route('/view_photo2',methods=['POST','GET'])
def view_photo2():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo2.html', result=value,vid=vid)    

@app.route('/view_photo3',methods=['POST','GET'])
def view_photo3():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo3.html', result=value,vid=vid)

@app.route('/view_photo4',methods=['POST','GET'])
def view_photo4():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo4.html', result=value,vid=vid)

@app.route('/message',methods=['POST','GET'])
def message():
    vid=""
    name=""
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT name FROM register where id=%s",(vid, ))
        name = mycursor.fetchone()[0]
    return render_template('message.html',vid=vid,name=name)


@app.route('/login',methods=['POST','GET'])
def login():
    uname=""
##    value=["1","2","3","4","5","6","7","8","9","0"]
##    change=random.shuffle(value)
##    print(change)
    if 'username' in session:
        uname = session['username']
    print(uname)
    mycursor1 = mydb.cursor()

    mycursor1.execute("SELECT * FROM register where card=%s",(uname, ))
    value = mycursor1.fetchone()
    accno=value[5]
    session['accno'] = accno
    
    mycursor1.execute("SELECT number FROM numbers order by rand()")
    value = mycursor1.fetchall()
    msg=""
        
    if request.method == 'POST':
        password1 = request.form['password']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM register where card=%s && password=%s",(uname, password1))
        myresult = mycursor.fetchone()[0]
        if password1=="":
            
            return render_template('login.html')
        else:
            
            #if str(password1)==str(myresult[10]):
            if myresult>0:
                #ff2=open("log.txt","w")
                #ff2.write(password1)
                #ff2.close()
                result=" Your Logged in sucessfully**"
                
                return redirect(url_for('userhome'))
            else:
                msg="Your logged in fail!!!"
                #return render_template('userhome.html',result=result)
    
    
    return render_template('login.html',value=value,msg=msg)



@app.route('/userhome')
def userhome():
    uname=""
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close() 

    name=""
    
   
    

    print(uname)
    mycursor1 = mydb.cursor()
    mycursor1.execute("SELECT * FROM register where card=%s",(uname, ))
    value = mycursor1.fetchone()
    print(value)
    name=value[1]  
        
    return render_template('userhome.html',name=name)

'''@app.route('/deposit')
def deposit():
    return render_template('deposit.html')
@app.route('/deposit_amount',methods=['POST','GET'])
def deposit_amount():
    if request.method=='POST':
        name=request.form['name']
        accountno=request.form['accno']
        amount=request.form['amount']
        today = date.today()
        rdate = today.strftime("%b-%d-%Y")
        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM event")
        maxid = mycursor.fetchone()[0]
        sql = "INSERT INTO event(id, name, accno, amount, rdate) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid, name, accountno, amount, rdate)
        mycursor.execute(sql, val)
        mydb.commit()   
    return render_template('userhome.html')'''

'''@app.route('/withdraw')
def withdraw():

    
    return render_template('withdraw.html')'''

@app.route('/verify_face',methods=['POST','GET'])
def verify_face():
    msg=""
    ss=""
    uname=""
    act=""
    if request.method=='GET':
        act = request.args.get('act')
        
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    print("uname="+uname)
    shutil.copy('faces/f1.jpg', 'static/f1.jpg')

    ff3=open("img.txt","r")
    mcnt=ff3.read()
    ff3.close()

    mcnt1=int(mcnt)
    if mcnt1==2:
        msg="Face Detected"
    elif mcnt1>2:
        msg="Multiple Face Detected!"
    else:
        msg="Face not Detected"
   
    
    
                
    return render_template('verify_face.html',msg=msg,act=act,mcnt1=mcnt1)

@app.route('/process',methods=['POST','GET'])
def process():
    vid=""
    pg="0"
    act="1"
    uname=""
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    value=[]
    shutil.copy('faces/f1.jpg', 'static/f1.jpg')
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM register WHERE card = %s', (uname, ))
    account = cursor.fetchone()
    name=account[1]
    mobile=account[3]
    
    email=account[4]
    vid=account[0]
    
    return render_template('process.html', vid=vid,pg=pg,act=act)

@app.route('/pro',methods=['POST','GET'])
def pro():
    vid=""
    value=[]
    pgg=0
    act="1"
    uname=""
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    if request.method=='GET':
        act = request.args.get('act')
    
        vid = request.args.get('vid')
        pg = request.args.get('pg')
        #pgg=int(pg)+1
        pgg=2
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM vt_face where vid=%s",(vid,))
        dtt = mycursor.fetchone()[0]
        
        if dtt<=pgg:
            act="1"
        else:
            act="2"
        
        mycursor.execute("SELECT vface FROM vt_face where vid=%s limit 0,1",(vid, ))
        value = mycursor.fetchone()[0]
        #print(value)
        
    return render_template('pro.html', result=value,vid=vid,pg=pgg,act=act)

@app.route('/verify_face2',methods=['POST','GET'])
def verify_face2():
    msg=""
    ss=""
    uname=""
    act=""
    if request.method=='GET':
        act = request.args.get('act')
        
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM register WHERE card = %s', (uname, ))
    account = cursor.fetchone()
    name=account[1]
    mobile=account[3]
    print(mobile)
    email=account[4]
    vid=account[0]
    
    
    shutil.copy('faces/f1.jpg', 'faces/s1.jpg')
    cutoff=5
    img="v"+str(vid)+".jpg"
    cursor.execute('SELECT * FROM vt_face WHERE vid = %s', (vid, ))
    dt = cursor.fetchall()
    for rr in dt:
        hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
        hash1 = imagehash.average_hash(Image.open("faces/s1.jpg"))
        cc1=hash0 - hash1
        print("cc="+str(cc1))
        if cc1<=cutoff:
            ss="ok"
            break
        else:
            ss="no"
    if ss=="ok":
        act="2"
        msg="Face Verified"
        print("correct person")
        return redirect(url_for('userhome', msg=msg))
    else:
        act="1"
        msg="Face not Verified"
        print("wrong person")
        #xn=randint(1000, 9999)
        #otp=str(xn)
        
        #cursor1 = mydb.cursor()
        #cursor1.execute('update register set otp=%s WHERE card = %s', (otp, uname))
        #mydb.commit()

        mess="Someone Access your account"
        url2="http://localhost/atm/img.txt"
        ur = urlopen(url2)#open url
        data1 = ur.read().decode('utf-8')

       
        idd=int(data1)+1
        url="http://iotcloud.co.in/testsms/sms.php?sms=link&name="+name+"&mess="+mess+"&mobile="+str(mobile)+"&id="+str(idd)
        print(url)
        webbrowser.open_new(url)
            
                
    return render_template('verify_face2.html',msg=msg,act=act)

@app.route('/cap',methods=['POST','GET'])
def cap():
    msg=""
    return render_template('cap.html',msg=msg)

@app.route('/verify',methods=['POST','GET'])
def verify():
    msg=""
    data1=""
    #act=""
    amtt=""
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    #data1="4"
    url2="http://localhost/atm/log.txt"
    ur = urlopen(url2)#open url
    data1 = ur.read().decode('utf-8')
    vv=data1.split('-')
    data1=vv[0]
    amtt=vv[1]
    print(data1)
    act = request.args.get('act')
    if act is None:
        act=""
    print("act="+str(act))
    if act=="3":
        amt=0
        amt1=0
        amt2=0
    
        
        amount1=amtt
        
        mycursor = mydb.cursor()

        mycursor.execute("SELECT amount FROM admin where username='admin'")
        amt1 = mycursor.fetchone()[0]

        mycursor.execute("SELECT deposit FROM register where card=%s",(uname, ))
        amt2 = mycursor.fetchone()[0]

        mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
        ddt = mycursor.fetchone()
        name=ddt[1]
        mobile=ddt[3]

        amt=int(amount1)
        if amt<=amt1:

            if amt<=amt2:
                mycursor.execute("UPDATE admin SET amount=amount-%s WHERE username='admin'",(amount1, ))
                mydb.commit()
                mycursor.execute("UPDATE register SET deposit=deposit-%s WHERE card=%s",(amount1, uname))
                mydb.commit()

                now = datetime.datetime.now()
                rdate=now.strftime("%d-%m-%Y")
                mycursor.execute("SELECT max(id)+1 FROM event")
                maxid = mycursor.fetchone()[0]
                if maxid is None:
                    maxid=1
                sql = "INSERT INTO event(id, name, accno, amount, rdate) VALUES (%s, %s, %s, %s, %s)"
                val = (maxid, name, uname, amt, rdate)
                mycursor.execute(sql, val)
                mydb.commit()

                mess="Amount Debited Rs."+str(amt)
                url="http://iotcloud.co.in/testsms/sms.php?sms=msg&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                webbrowser.open_new(url)
            
                msg="Withdraw success..."
            else:
                mess="Your Account balance is low!"
                url="http://iotcloud.co.in/testsms/sms.php?sms=msg&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                webbrowser.open_new(url)
                msg="Your Account balance is low!"
        else:
            msg="Cash is not available in ATM!!"
    
        
    return render_template('verify.html',msg=msg,act=act,amtt=amtt,data1=data1)


@app.route('/otp', methods=['GET', 'POST'])
def otp():
    msg=""
    key=""
    if 'username' in session:
        uname = session['username']
    cursor = mydb.cursor()
    cursor.execute('SELECT otp FROM register WHERE card = %s', (uname, ))
    account = cursor.fetchone()[0]
    key=account
    
    if request.method=='POST':
        otp=request.form['otp']
        
        if otp==key:
            session['username'] = uname
            
            return redirect(url_for('verify_aadhar'))
        else:
            msg = 'OTP wrong!'
    return render_template('otp.html',msg=msg,key=key)

@app.route('/atm_balance',methods=['POST','GET'])
def atm_balance():
    msg=""
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()

    cursor = mydb.cursor()
    if request.method=='POST':
        amount=request.form['amount']
        cursor.execute("UPDATE admin SET amount=%s WHERE username='admin'",(amount, ))
        mydb.commit()
        return redirect(url_for('admin'))

        
    
    cursor.execute("SELECT amount FROM admin WHERE username='admin'")
    value = cursor.fetchone()[0]
    
    return render_template('atm_balance.html',msg=msg,value=value)

@app.route('/withdraw',methods=['POST','GET'])
def withdraw():
    uname=""
    ##if 'username' in session:
    #    uname = session['username']
    #    accno = session['accno']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close() 
    msg=""
    amt=0
    amt1=0
    amt2=0
    if request.method=='POST':
        
        amount1=request.form['amount']
        
        mycursor = mydb.cursor()

        mycursor.execute("SELECT amount FROM admin where username='admin'")
        amt1 = mycursor.fetchone()[0]

        mycursor.execute("SELECT deposit FROM register where card=%s",(uname, ))
        amt2 = mycursor.fetchone()[0]

        mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
        ddt = mycursor.fetchone()
        name=ddt[1]
        mobile=ddt[3]

        amt=int(amount1)
        if amt<=amt1:

            if amt<=amt2:
                mycursor.execute("UPDATE admin SET amount=amount-%s WHERE username='admin'",(amount1, ))
                mydb.commit()
                mycursor.execute("UPDATE register SET deposit=deposit-%s WHERE card=%s",(amount1, uname))
                mydb.commit()

                now = datetime.datetime.now()
                rdate=now.strftime("%d-%m-%Y")
                mycursor.execute("SELECT max(id)+1 FROM event")
                maxid = mycursor.fetchone()[0]
                if maxid is None:
                    maxid=1
                sql = "INSERT INTO event(id, name, accno, amount, rdate) VALUES (%s, %s, %s, %s, %s)"
                val = (maxid, name, uname, amt, rdate)
                mycursor.execute(sql, val)
                mydb.commit()

                mess="Amount Debited Rs."+str(amt)
                url="http://iotcloud.co.in/testsms/sms.php?sms=msg&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                webbrowser.open_new(url)
            
                msg="Withdraw success..."
            else:
                msg="Your Account balance is low!"
        else:
            msg="Cash is not available in ATM!!"
        
    return render_template('withdraw.html',msg=msg)


@app.route('/balance')
def balance():
    uname=""
    #if 'username' in session:
    #    uname = session['username']
    #    accno = session['accno']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close() 
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
    data = mycursor.fetchone()
    deposit=data[9]
    print(str(deposit))
    return render_template('balance.html', data=deposit)



@app.route('/user_view')
def user_view():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register")
    result = mycursor.fetchall()
    return render_template('user_view.html', result=result)

@app.route('/view_withdraw')
def view_withdraw():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM event order by id desc")
    result = mycursor.fetchall()
    return render_template('view_withdraw.html', result=result)

@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    #session.pop('username', None)
    return redirect(url_for('index'))

def gen(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
        

def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
