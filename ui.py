import random
import os
import time
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog, messagebox, StringVar, Entry, Label, Button
from PIL import Image, ImageTk
from siamese import Siamese
from ultralytics import YOLO
import yaml

def updata_yaml(k,v):
    old_data=yaml.load(open("ultralytics\cfg\datasets\my.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    old_data[k]=v #修改读取的数据（k存在就修改对应值，k不存在就新增一组键值对）
    with open("ultralytics\cfg\datasets\my.yaml", "w", encoding="utf-8") as f:
        yaml.dump(old_data,f)
class WinGUI(Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.gtsb_srk1 = StringVar()
        self.gtsb_srk2 = StringVar()
        self.gtsb_srk3 = StringVar()
        self.plgtsb_srk1 = StringVar()
        self.plgtsb_srk2 = StringVar()
        self.zlyc_srk_1 = StringVar()
        self.plzlyc_srk_1 = StringVar()
        self.tk_tabs_xxk = self.__tk_tabs_xxk(self)
        self.tk_label_lyoao1t4 = self.__tk_label_lyoao1t4( self.tk_tabs_xxk_2)
        self.tk_label_lyoao3mr = self.__tk_label_lyoao3mr( self.tk_tabs_xxk_2)
        self.tk_input_gtsb_srk_1 = self.__tk_input_gtsb_srk_1( self.tk_tabs_xxk_2, self.gtsb_srk1)
        self.tk_input_gtsb_srk_2 = self.__tk_input_gtsb_srk_2( self.tk_tabs_xxk_2, self.gtsb_srk2)
        self.tk_input_gtsb_srk_3 = self.__tk_input_gtsb_srk_3( self.tk_tabs_xxk_2, self.gtsb_srk3)
        self.tk_button_gtsb_jztp1 = self.__tk_button_gtsb_jztp1( self.tk_tabs_xxk_2)
        self.tk_button_gtsb_jztp2 = self.__tk_button_gtsb_jztp2( self.tk_tabs_xxk_2)
        self.tk_button_jzmx1 = self.__tk_button_jzmx1( self.tk_tabs_xxk_4)
        self.tk_button_jzmx2 = self.__tk_button_jzmx2( self.tk_tabs_xxk_4)
        self.tk_button_gtsb_yc = self.__tk_button_gtsb_yc( self.tk_tabs_xxk_2)
        self.tk_label_gtsb_hb_1 = self.__tk_label_gtsb_hb_1( self.tk_tabs_xxk_2)
        self.tk_label_gtsb_hb_2 = self.__tk_label_gtsb_hb_2( self.tk_tabs_xxk_2)
        self.tk_input_plgtsb_srk1 = self.__tk_input_plgtsb_srk1( self.tk_tabs_xxk_3, self.plgtsb_srk1)
        self.tk_input_plgtsb_srk2 = self.__tk_input_plgtsb_srk2( self.tk_tabs_xxk_3, self.plgtsb_srk2)
        self.tk_label_lyoaxbal = self.__tk_label_lyoaxbal( self.tk_tabs_xxk_2)
        self.tk_label_lyob2c2w = self.__tk_label_lyob2c2w( self.tk_tabs_xxk_3)
        self.tk_label_lyob2epo = self.__tk_label_lyob2epo( self.tk_tabs_xxk_3)
        self.tk_button_lyob2q6t = self.__tk_button_lyob2q6t( self.tk_tabs_xxk_3)
        self.tk_button_lyob2rlg = self.__tk_button_lyob2rlg( self.tk_tabs_xxk_3)
        self.tk_button_lyob5diw = self.__tk_button_lyob5diw( self.tk_tabs_xxk_3)
        self.tk_progressbar_lyobe33h = self.__tk_progressbar_lyobe33h( self.tk_tabs_xxk_3)
        self.tk_text_plgtsb_wbk = self.__tk_text_plgtsb_wbk( self.tk_tabs_xxk_3)
        self.tk_label_zlyc_bq1 = self.__tk_label_zlyc_bq1( self.tk_tabs_xxk_0)
        self.tk_input_zlyc_srk_1 = self.__tk_input_zlyc_srk_1( self.tk_tabs_xxk_0,self.zlyc_srk_1)
        self.tk_button_zlyc_xztp = self.__tk_button_zlyc_xztp( self.tk_tabs_xxk_0)
        self.tk_button_zlyc_xzwj = self.__tk_button_zlyc_xzwj( self.tk_tabs_xxk_0)
        self.tk_button_zlyc_yc = self.__tk_button_zlyc_yc( self.tk_tabs_xxk_0)
        self.tk_text_zlyc_hb_1 = self.__tk_text_zlyc_hb_1( self.tk_tabs_xxk_0)
        self.tk_text_zlyc_wbk_1 = self.__tk_text_zlyc_wbk_1( self.tk_tabs_xxk_0)
        self.tk_label_plzlyc_bq_1 = self.__tk_label_plzlyc_bq_1( self.tk_tabs_xxk_1)
        self.tk_input_plzlyc_srk_1 = self.__tk_input_plzlyc_srk_1( self.tk_tabs_xxk_1, self.plzlyc_srk_1)
        self.tk_button_plzlyc_xzwjj = self.__tk_button_plzlyc_xzwjj( self.tk_tabs_xxk_1)
        self.tk_button_plzlyc_yc = self.__tk_button_plzlyc_yc( self.tk_tabs_xxk_1)
        self.tk_text_plzlyc_wbk_1 = self.__tk_text_plzlyc_wbk_1( self.tk_tabs_xxk_1)
    def __win(self):
        self.title("识别的都队模型UI")
        # 设置窗口大小、居中
        width = 700
        height = 500
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        
        self.resizable(width=False, height=False)
        
    def scrollbar_autohide(self,vbar, hbar, widget):
        """自动隐藏滚动条"""
        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)
        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)
        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())
    
    def v_scrollbar(self,vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')
    def h_scrollbar(self,hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')
    def create_bar(self,master, widget,is_vbar,is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)
    def __tk_tabs_xxk(self,parent):
        frame = Notebook(parent)
        self.tk_tabs_xxk_4 = self.__tk_frame_xxk_4(frame)
        frame.add(self.tk_tabs_xxk_4, text="模型加载")
        self.tk_tabs_xxk_0 = self.__tk_frame_xxk_0(frame)
        frame.add(self.tk_tabs_xxk_0, text="种类识别")
        self.tk_tabs_xxk_1 = self.__tk_frame_xxk_1(frame)
        frame.add(self.tk_tabs_xxk_1, text="批量种类识别")
        self.tk_tabs_xxk_2 = self.__tk_frame_xxk_2(frame)
        frame.add(self.tk_tabs_xxk_2, text="  个体识别  ")
        self.tk_tabs_xxk_3 = self.__tk_frame_xxk_3(frame)
        frame.add(self.tk_tabs_xxk_3, text="  批量个体识别  ")
        frame.place(x=0, y=0, width=700, height=500)
        return frame
    def __tk_frame_xxk_4(self,parent):
        frame = Frame(parent)
        frame.place(x=0, y=0, width=700, height=500)
        return frame
    def __tk_frame_xxk_0(self,parent):
        frame = Frame(parent)
        frame.place(x=0, y=0, width=700, height=500)
        return frame
    def __tk_frame_xxk_1(self,parent):
        frame = Frame(parent)
        frame.place(x=0, y=0, width=700, height=500)
        return frame
    def __tk_frame_xxk_2(self,parent):
        frame = Frame(parent)
        frame.place(x=0, y=0, width=700, height=500)
        return frame
    def __tk_frame_xxk_3(self,parent):
        frame = Frame(parent)
        frame.place(x=0, y=0, width=700, height=500)
        return frame
    def __tk_label_lyoao1t4(self,parent):
        label = Label(parent,text="图片1：",anchor="center", )
        label.place(x=20, y=15, width=80, height=30)
        return label
    def __tk_label_lyoao3mr(self,parent):
        label = Label(parent,text="图片2：",anchor="center", )
        label.place(x=20, y=60, width=80, height=30)
        return label
    def __tk_input_gtsb_srk_1(self,parent,gtsb_srk_1):
        ipt = Entry(parent, textvariable=gtsb_srk_1)
        ipt.place(x=120, y=15, width=450, height=30)
        return ipt
    def __tk_input_gtsb_srk_2(self,parent,gtsb_srk_2):
        ipt = Entry(parent, textvariable=gtsb_srk_2)
        ipt.place(x=120, y=60, width=450, height=30)
        return ipt
    def __tk_input_gtsb_srk_3(self,parent ,gtsb_srk_3):
        ipt = Entry(parent,  textvariable=gtsb_srk_3)
        ipt.place(x=277, y=120, width=180, height=30)
        return ipt
    def __tk_input_plgtsb_srk1(self,parent,plgtsb_srk1):
        ipt = Entry(parent, textvariable=plgtsb_srk1 )
        ipt.place(x=150, y=40, width=400, height=30)
        return ipt
    def __tk_input_plgtsb_srk2(self,parent,plgtsb_srk2):
        ipt = Entry(parent,textvariable=plgtsb_srk2 )
        ipt.place(x=150, y=140, width=400, height=30)
        return ipt
    def __tk_button_gtsb_jztp1(self,parent):
        #按下后抬起事件
        btn = Button(parent, text="加载图片1", takefocus=False,)
        btn.place(x=600, y=15, width=80, height=30)
        return btn
    def __tk_button_gtsb_jztp2(self,parent):
        btn = Button(parent, text="加载图片2", takefocus=False,)
        btn.place(x=600, y=60, width=80, height=30)
        return btn
    def __tk_button_jzmx1(self,parent):
        btn = Button(parent, text="加载个体识别模型", takefocus=False,)
        btn.place(x=180, y=200, width=220, height=60)
        return btn
    def __tk_button_jzmx2(self,parent):
        btn = Button(parent, text="加载种类识别模型", takefocus=False,)
        btn.place(x=180, y=30, width=220, height=60)

        return btn
    def __tk_button_gtsb_yc(self,parent):
        btn = Button(parent, text="预测", takefocus=False,)
        btn.place(x=500, y=110, width=80, height=50)
        return btn
    def __tk_label_gtsb_hb_1(self, parent):
        # 创建一个 Label 用于显示图片1
        self.tk_label_gtsb_hb_1 = Label(parent, bg="#aaa")
        self.tk_label_gtsb_hb_1.place(x=50, y=180, width=250, height=250)
        return self.tk_label_gtsb_hb_1

    def __tk_label_gtsb_hb_2(self, parent):
        # 创建一个 Label 用于显示图片2
        self.tk_label_gtsb_hb_2 = Label(parent, bg="#aaa")
        self.tk_label_gtsb_hb_2.place(x=400, y=180, width=250, height=250)
        return self.tk_label_gtsb_hb_2

    
    def __tk_label_lyoaxbal(self,parent):
        label = Label(parent,text="预测结果：",anchor="center", )
        label.place(x=190, y=120, width=60, height=30)
        return label
    def __tk_label_lyob2c2w(self,parent):
        label = Label(parent,text="个体识别图片文件夹",anchor="center", )
        label.place(x=10, y=40, width=120, height=30)
        return label
    def __tk_label_lyob2epo(self,parent):
        label = Label(parent,text="个体识别标签",anchor="center", )
        label.place(x=10, y=140, width=120, height=30)
        return label
    
    def __tk_button_lyob2q6t(self,parent):
        btn = Button(parent, text="选择文件夹", takefocus=False,)
        btn.place(x=570, y=40, width=100, height=30)
        return btn
    def __tk_button_lyob2rlg(self,parent):
        btn = Button(parent, text="加载文件", takefocus=False,)
        btn.place(x=570, y=140, width=100, height=30)
        return btn
    def __tk_button_lyob5diw(self,parent):
        btn = Button(parent, text="预测", takefocus=False,)
        btn.place(x=300, y=200, width=125, height=70)
        return btn
    def __tk_progressbar_lyobe33h(self,parent):
        progressbar = Progressbar(parent, orient=HORIZONTAL,)
        progressbar.place(x=70, y=290, width=550, height=30)
        progressbar['maximum'] = 100
        progressbar['value'] = 0
        return progressbar
    def __tk_text_plgtsb_wbk(self,parent):
        text = Text(parent)
        text.place(x=70, y=350, width=550, height=100)
        return text
    def __tk_label_zlyc_bq1(self,parent):
        label = Label(parent,text="预测图片：",anchor="center",)
        label.place(x=50, y=15, width=70, height=50)
        return label
    

    def __tk_input_zlyc_srk_1(self,parent,zlyc_srk_1):
        ipt = Entry(parent, textvariable=zlyc_srk_1 )
        ipt.place(x=150, y=15, width=360, height=50)
        return ipt
    def __tk_button_zlyc_xztp(self,parent):
        btn = Button(parent, text="选择图片", takefocus=False,)
        btn.place(x=520, y=20, width=60, height=40)
        return btn
    def __tk_button_zlyc_xzwj(self,parent):
        btn = Button(parent, text="选择文件夹", takefocus=False,)
        btn.place(x=600, y=20, width=70, height=40)
        return btn
    def __tk_button_zlyc_yc(self,parent):
        #文字大小
        btn = Button(parent, text="预测", takefocus=False, font=('Arial', 20))
        btn.place(x=428, y=290, width=250, height=140)
        return btn
    def __tk_text_zlyc_hb_1(self,parent):
        text = Text(parent)
        text.place(x=35, y=85, width=350, height=350)
        return text
    def __tk_text_zlyc_wbk_1(self,parent):
        text = Text(parent)
        text.place(x=425, y=85, width=250, height=170)
        return text
    def __tk_label_plzlyc_bq_1(self,parent):
        label = Label(parent,text="预测文件夹：",anchor="center", )
        label.place(x=15, y=20, width=80, height=30)
        return label
    def __tk_input_plzlyc_srk_1(self,parent,plzlyc_srk_1):
        ipt = Entry(parent, textvariable=plzlyc_srk_1)
        ipt.place(x=110, y=20, width=450, height=30)
        return ipt
    def __tk_button_plzlyc_xzwjj(self,parent):
        btn = Button(parent, text="选择文件夹", takefocus=False,)
        btn.place(x=580, y=20, width=100, height=30)
        return btn
    def __tk_button_plzlyc_yc(self,parent):
        btn = Button(parent, text="预测", takefocus=False,)
        btn.place(x=270, y=80, width=180, height=60)
        return btn
    def __tk_text_plzlyc_wbk_1(self,parent):
        text = Text(parent)
        text.place(x=169, y=170, width=382, height=180)
        return text

class Win(WinGUI):
    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.__style_config()
        self.ctl.init(self)

    def __event_bind(self):
        self.tk_button_gtsb_jztp1.bind('<Button>',self.ctl.gtsb_jztp_1)
        self.tk_button_gtsb_jztp2.bind('<Button>',self.ctl.gtsb_jztp_2)
        self.tk_button_gtsb_yc.bind('<Button>',self.ctl.gtsb_yc)
        self.tk_button_lyob2q6t.bind('<Button>',self.ctl.plgtsb_xzwjj)
        self.tk_button_lyob2rlg.bind('<Button>',self.ctl.plgtsb_jzwj)
        self.tk_button_lyob5diw.bind('<Button>',self.ctl.plgtsb_yc)
        self.tk_button_zlyc_xztp.bind('<Button>',self.ctl.zlyc_jztp)
        self.tk_button_zlyc_xzwj.bind('<Button>',self.ctl.zlyc_jzwj)
        self.tk_button_zlyc_yc.bind('<Button>',self.ctl.zlyc_yc)
        self.tk_button_plzlyc_xzwjj.bind('<Button>',self.ctl.plzlyc_xzwjj)
        self.tk_button_plzlyc_yc.bind('<Button>',self.ctl.plzlyc_yc)
        self.tk_button_jzmx1.bind('<Button>',self.ctl.jzmx1)
        self.tk_button_jzmx2.bind('<Button>',self.ctl.jzmx2)
        pass
    def __style_config(self):
        pass


class Controller:
    ui: object

    def __init__(self):
        pass
    def init(self, ui):
        self.ui = ui
        # TODO 组件初始化 赋值操作
    def jzmx1(self,evt):
        # 加载个体识别模型
        file_path = filedialog.askopenfilename()
        if file_path:
            self.model = Siamese(model_path=file_path)
            self.model.net.eval()
            #随机生成两个图片进行一次预测，（ps第一次预测慢）
            image_1 = Image.new('RGB', (224, 224), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            image_2 = Image.new('RGB', (224, 224), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            self.model.detect_image(image_1, image_2)
            messagebox.showinfo("Success", "个体识别模型导入成功.")
        else:
            messagebox.showinfo("Success", "个体识别模型导入失败，请重新导入.")
    def jzmx2(self,evt):
        # 加载种类识别模型
        file_path = filedialog.askopenfilename()
        if file_path:
            self.model_classify = YOLO(file_path)
            image = Image.new('RGB', (224, 224), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            self.model_classify(image)
            messagebox.showinfo("Success", "种类识别模型导入成功.")
        else:
            messagebox.showinfo("Success", "种类识别模型导入失败，请重新导入.")
    def gtsb_jztp_1(self, evt):
        # 加载图片1并显示在Canvas上
        file_path = filedialog.askopenfilename()
        if file_path:
            self.ui.gtsb_srk1.set(file_path)
            # 使用PIL打开图片
            image = Image.open(file_path)
            image_resized = image.resize((250, 250), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image_resized)
            self.ui.tk_label_gtsb_hb_1.config(image=photo)
            #刷新显示
            self.ui.tk_label_gtsb_hb_1.image = photo

    def gtsb_jztp_2(self, evt):
        # 加载图片2并显示在指定位置上
        file_path = filedialog.askopenfilename()
        if file_path:
            self.ui.gtsb_srk2.set(file_path)
            image = Image.open(file_path)
            image_resized = image.resize((250, 250), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image_resized)
            self.ui.tk_label_gtsb_hb_2.config(image=photo)
            self.ui.tk_label_gtsb_hb_2.image = photo
  
    def gtsb_yc(self,evt):
        # 预测
        #清空原有的预测结果
        self.ui.gtsb_srk3.set("")
        #记录预测开始时间
        time_start = time.time()
        image_1_path = self.ui.gtsb_srk1.get()
        image_2_path = self.ui.gtsb_srk2.get()
        if not image_1_path or not image_2_path:
            messagebox.showerror("Error", "Please load images first.")
            return
        # TODO 预测逻辑
        try:
            image_1 = Image.open(image_1_path)
            image_2 = Image.open(image_2_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error opening images: {str(e)}")
            return
        
        probability = self.model.detect_image(image_1, image_2)
        print(probability)
        time_end = time.time()
        elapsed_time_ms = (time_end - time_start) * 1000
        formatted_string = f"{probability}   耗时: {elapsed_time_ms:.2f}ms"
        self.ui.gtsb_srk3.set(formatted_string)

        #记录预测结束时间
        
    def plgtsb_xzwjj(self,evt):
        # 选择文件夹
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.ui.plgtsb_srk1.set(folder_path)
    def plgtsb_jzwj(self,evt):
        #选择.txt文件
        file_path = filedialog.askopenfilename()
        if file_path:
            self.ui.plgtsb_srk2.set(file_path)
    def plgtsb_yc(self,evt):
        # 批量预测
        pic_path = self.ui.plgtsb_srk1.get()
        file_path = self.ui.plgtsb_srk2.get()
        l = []
        t = []
        count = 0
        num=0
        with open(file_path, 'r') as f:
            #文本框显示预测中
            self.ui.tk_text_plgtsb_wbk.delete(1.0, END)
            self.ui.tk_text_plgtsb_wbk.insert(END, "预测中...")
            lines = f.readlines()
            
            for line in lines:
                #进度条显示预测进度
                self.ui.tk_progressbar_lyobe33h['value'] = count / len(lines) * 100
                #print(count / len(lines) * 100)
                self.ui.tk_progressbar_lyobe33h.update()
                #更新整个窗口
                #self.ui.update()

                #加载图片
                img1 = Image.open(os.path.join(pic_path, line.split(',')[0]))
                img2 = Image.open(os.path.join(pic_path, line.split(',')[1]))
                title = line.split(',')[2]

                result = self.model.detect_image(img1, img2)
                #print(i+1,result, title)
                l.append(bool(result))
                t.append(title)
                count += 1
                if (result == True and title == 'True\n') or (result == False and title == 'False\n'):
                    num += 1
        #将预测结果写入.txt文件 文件名+预测结果+标签
        #文本框显示将预测结果写入result.txt
        self.ui.tk_text_plgtsb_wbk.delete(1.0, END)
        self.ui.tk_text_plgtsb_wbk.insert(END, "将预测结果写入result.txt")
        
        with open('result.txt', 'w') as f:
            for i in range(len(l)):
                f.write(str(i+1) + ',' + str(l[i]) + ',' + t[i])
        #显示完成
        self.ui.tk_text_plgtsb_wbk.delete(1.0, END)
        self.ui.tk_text_plgtsb_wbk.insert(END, "预测完成,预测结果保存在result.txt中")
        self.ui.tk_text_plgtsb_wbk.insert(END, "共预测" + str(len(l)) + "个样本，")
        self.ui.tk_text_plgtsb_wbk.insert(END, "准确率: " + str(num / len(l)))
    def zlyc_jztp(self,evt):
        # 选择图片
        file_path = filedialog.askopenfilename()
        if file_path:
            self.ui.zlyc_srk_1.set(file_path)
            image = Image.open(file_path)
            image_resized = image.resize((350, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image_resized)
            self.ui.tk_text_zlyc_hb_1.delete(1.0, END)
            self.ui.tk_text_zlyc_hb_1.image_create(END, image=photo)
            self.ui.tk_text_zlyc_hb_1.image = photo
    def zlyc_jzwj(self,evt):
        # 选择文件夹
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.ui.zlyc_srk_1.set(folder_path)
    def zlyc_yc(self,evt):
        # 预测
        file_path = self.ui.zlyc_srk_1.get()
        if not file_path:
            messagebox.showerror("Error", "Please load image first.")
            return
        # try:
        #     image = Image.open(file_path)
        # except Exception as e:
        #     messagebox.showerror("Error", f"Error opening image: {str(e)}")
        #     return
        # 预测
        results = self.model_classify(file_path)
        self.ui.tk_text_zlyc_wbk_1.delete(1.0, END)
        for result in results:
            probs = result.probs  # Probs object for classification outputs
            print(result.names[probs.top1])
            self.ui.tk_text_zlyc_wbk_1.insert(END, result.names[probs.top1])
            self.ui.tk_text_zlyc_wbk_1.insert(END, '\n')
            #result.show()
    def plzlyc_xzwjj(self,evt):
        # 选择文件夹
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.ui.plzlyc_srk_1.set(folder_path)

    def plzlyc_yc(self,evt):
        # 批量预测
        pic_path = self.ui.plzlyc_srk_1.get()
        #修改yaml文件中的data字段
        updata_yaml("text",pic_path)
        self.ui.tk_text_plzlyc_wbk_1.delete(1.0, END)
        self.ui.tk_text_plzlyc_wbk_1.insert(END, "预测中...")
        self.ui.tk_text_plzlyc_wbk_1.update()
        self.model_classify.val(data=r'my.yaml',
              split='test',
              imgsz=224,
              batch=16,
              # iou=0.7,
              # rect=False,
              project='runs/val',
              name='exp',
              )
        self.ui.tk_text_plzlyc_wbk_1.delete(1.0, END)
        self.ui.tk_text_plzlyc_wbk_1.insert(END, "预测完成,预测结果保存在runs/val中")
        


app = Win(Controller())
if __name__ == "__main__":
    # 启动
    app.mainloop()