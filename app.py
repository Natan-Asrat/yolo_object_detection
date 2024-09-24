import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import math
import cvzone
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, RTCConfiguration
import av

# Custom Video Processor Class
class CustomVideoProcessor(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")  # Convert to OpenCV format
        results = self.model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil(box.conf[0] * 100) / 100

                # Draw the bounding box
                cvzone.cornerRect(img, (x1, y1, x2, y2))
                label = f"{self.model.names[int(box.cls[0])]} {confidence:.2f}"
                font_scale = 1.0  # Smaller font size
                thickness = 2     # Thinner text
                cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=font_scale, thickness=thickness)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main application logic
st.title("YOLO Object Detection Live Stream")
st.text("Live object detection by Natan Asrat.")

# Load the YOLO model
@st.cache_resource(show_spinner=True)
def load_model():
    st.text("Loading YOLO model...")
    model = YOLO('./yolo_weights/yolov8n.pt')
    return model

model = load_model()
rtc_configuration = RTCConfiguration({
    "iceServers": [
        { "urls": "stun:23.21.150.121:3478" },
        { "urls": "stun:iphone-stun.strato-iphone.de:3478" },
        { "urls": "stun:numb.viagenie.ca:3478" },
        { "urls": "stun:s1.taraba.net:3478" },
        { "urls": "stun:s2.taraba.net:3478" },
        { "urls": "stun:stun.12connect.com:3478" },
        { "urls": "stun:stun.12voip.com:3478" },
        { "urls": "stun:stun.1und1.de:3478" },
        { "urls": "stun:stun.2talk.co.nz:3478" },
        { "urls": "stun:stun.2talk.com:3478" },
        { "urls": "stun:stun.3clogic.com:3478" },
        { "urls": "stun:stun.3cx.com:3478" },
        { "urls": "stun:stun.a-mm.tv:3478" },
        { "urls": "stun:stun.aa.net.uk:3478" },
        { "urls": "stun:stun.acrobits.cz:3478" },
        { "urls": "stun:stun.actionvoip.com:3478" },
        { "urls": "stun:stun.advfn.com:3478" },
        { "urls": "stun:stun.aeta-audio.com:3478" },
        { "urls": "stun:stun.aeta.com:3478" },
        { "urls": "stun:stun.alltel.com.au:3478" },
        { "urls": "stun:stun.altar.com.pl:3478" },
        { "urls": "stun:stun.annatel.net:3478" },
        { "urls": "stun:stun.antisip.com:3478" },
        { "urls": "stun:stun.arbuz.ru:3478" },
        { "urls": "stun:stun.avigora.com:3478" },
        { "urls": "stun:stun.avigora.fr:3478" },
        { "urls": "stun:stun.awa-shima.com:3478" },
        { "urls": "stun:stun.awt.be:3478" },
        { "urls": "stun:stun.b2b2c.ca:3478" },
        { "urls": "stun:stun.bahnhof.net:3478" },
        { "urls": "stun:stun.barracuda.com:3478" },
        { "urls": "stun:stun.bluesip.net:3478" },
        { "urls": "stun:stun.bmwgs.cz:3478" },
        { "urls": "stun:stun.botonakis.com:3478" },
        { "urls": "stun:stun.budgetphone.nl:3478" },
        { "urls": "stun:stun.budgetsip.com:3478" },
        { "urls": "stun:stun.cablenet-as.net:3478" },
        { "urls": "stun:stun.callromania.ro:3478" },
        { "urls": "stun:stun.callwithus.com:3478" },
        { "urls": "stun:stun.cbsys.net:3478" },
        { "urls": "stun:stun.chathelp.ru:3478" },
        { "urls": "stun:stun.cheapvoip.com:3478" },
        { "urls": "stun:stun.ciktel.com:3478" },
        { "urls": "stun:stun.cloopen.com:3478" },
        { "urls": "stun:stun.colouredlines.com.au:3478" },
        { "urls": "stun:stun.comfi.com:3478" },
        { "urls": "stun:stun.commpeak.com:3478" },
        { "urls": "stun:stun.comtube.com:3478" },
        { "urls": "stun:stun.comtube.ru:3478" },
        { "urls": "stun:stun.cope.es:3478" },
        { "urls": "stun:stun.counterpath.com:3478" },
        { "urls": "stun:stun.counterpath.net:3478" },
        { "urls": "stun:stun.cryptonit.net:3478" },
        { "urls": "stun:stun.darioflaccovio.it:3478" },
        { "urls": "stun:stun.datamanagement.it:3478" },
        { "urls": "stun:stun.dcalling.de:3478" },
        { "urls": "stun:stun.decanet.fr:3478" },
        { "urls": "stun:stun.demos.ru:3478" },
        { "urls": "stun:stun.develz.org:3478" },
        { "urls": "stun:stun.dingaling.ca:3478" },
        { "urls": "stun:stun.doublerobotics.com:3478" },
        { "urls": "stun:stun.drogon.net:3478" },
        { "urls": "stun:stun.duocom.es:3478" },
        { "urls": "stun:stun.dus.net:3478" },
        { "urls": "stun:stun.e-fon.ch:3478" },
        { "urls": "stun:stun.easybell.de:3478" },
        { "urls": "stun:stun.easycall.pl:3478" },
        { "urls": "stun:stun.easyvoip.com:3478" },
        { "urls": "stun:stun.efficace-factory.com:3478" },
        { "urls": "stun:stun.einsundeins.com:3478" },
        { "urls": "stun:stun.einsundeins.de:3478" },
        { "urls": "stun:stun.ekiga.net:3478" },
        { "urls": "stun:stun.epygi.com:3478" },
        { "urls": "stun:stun.etoilediese.fr:3478" },
        { "urls": "stun:stun.eyeball.com:3478" },
        { "urls": "stun:stun.faktortel.com.au:3478" },
        { "urls": "stun:stun.freecall.com:3478" },
        { "urls": "stun:stun.freeswitch.org:3478" },
        { "urls": "stun:stun.freevoipdeal.com:3478" },
        { "urls": "stun:stun.fuzemeeting.com:3478" },
        { "urls": "stun:stun.gmx.de:3478" },
        { "urls": "stun:stun.gmx.net:3478" },
        { "urls": "stun:stun.gradwell.com:3478" },
        { "urls": "stun:stun.halonet.pl:3478" },
        { "urls": "stun:stun.hellonanu.com:3478" },
        { "urls": "stun:stun.hoiio.com:3478" },
        { "urls": "stun:stun.hosteurope.de:3478" },
        { "urls": "stun:stun.ideasip.com:3478" },
        { "urls": "stun:stun.imesh.com:3478" },
        { "urls": "stun:stun.infra.net:3478" },
        { "urls": "stun:stun.internetcalls.com:3478" },
        { "urls": "stun:stun.intervoip.com:3478" },
        { "urls": "stun:stun.ipcomms.net:3478" },
        { "urls": "stun:stun.ipfire.org:3478" },
        { "urls": "stun:stun.ippi.fr:3478" },
        { "urls": "stun:stun.ipshka.com:3478" },
        { "urls": "stun:stun.iptel.org:3478" },
        { "urls": "stun:stun.irian.at:3478" },
        { "urls": "stun:stun.it1.hr:3478" },
        { "urls": "stun:stun.ivao.aero:3478" },
        { "urls": "stun:stun.jive.com:3478" },
        { "urls": "stun:stun.kaico.com:3478" },
        { "urls": "stun:stun.karsshop.com:3478" },
        { "urls": "stun:stun.kurtis.town:3478" },
        { "urls": "stun:stun.liberateip.com:3478" },
        { "urls": "stun:stun.lifesizecloud.com:3478" },
        { "urls": "stun:stun.lucytele.com:3478" },
        { "urls": "stun:stun.magma.com:3478" },
        { "urls": "stun:stun.marthasvineyard.net:3478" },
        { "urls": "stun:stun.mikrotik.com:3478" },
        { "urls": "stun:stun.miratelecom.com:3478" },
        { "urls": "stun:stun.mostafa.com:3478" },
        { "urls": "stun:stun.muscleme.com:3478" },
        { "urls": "stun:stun.netvoip.com:3478" },
        { "urls": "stun:stun.netvoip.eu:3478" },
        { "urls": "stun:stun.netvoip.net:3478" },
        { "urls": "stun:stun.nline.net:3478" },
        { "urls": "stun:stun.nordvpn.com:3478" },
        { "urls": "stun:stun.nortel.com:3478" },
        { "urls": "stun:stun.novatel.ca:3478" },
        { "urls": "stun:stun.novatel.co.uk:3478" },
        { "urls": "stun:stun.oxer.com:3478" },
        { "urls": "stun:stun.ozvoip.com.au:3478" },
        { "urls": "stun:stun.pantel.pl:3478" },
        { "urls": "stun:stun.pennyvoip.com:3478" },
        { "urls": "stun:stun.pensum.com:3478" },
        { "urls": "stun:stun.plainvoip.com:3478" },
        { "urls": "stun:stun.popvox.com:3478" },
        { "urls": "stun:stun.purevoip.com:3478" },
        { "urls": "stun:stun.qcall.ru:3478" },
        { "urls": "stun:stun.raketu.com:3478" },
        { "urls": "stun:stun.rebtel.com:3478" },
        { "urls": "stun:stun.rebtel.net:3478" },
        { "urls": "stun:stun.ringcentral.com:3478" },
        { "urls": "stun:stun.ringcentral.net:3478" },
        { "urls": "stun:stun.sip.kuwait.net:3478" },
        { "urls": "stun:stun.sipcall.com:3478" },
        { "urls": "stun:stun.sipgate.de:3478" },
        { "urls": "stun:stun.sipgate.com:3478" },
        { "urls": "stun:stun.sipline.de:3478" },
        { "urls": "stun:stun.sipnet.com:3478" },
        { "urls": "stun:stun.sipnetwork.eu:3478" },
        { "urls": "stun:stun.sipphone.com:3478" },
        { "urls": "stun:stun.siptrunk.com:3478" },
        { "urls": "stun:stun.sipgate.com.au:3478" },
        { "urls": "stun:stun.sipvoice.com:3478" },
        { "urls": "stun:stun.smartvoice.eu:3478" },
        { "urls": "stun:stun.sipwitch.com:3478" },
        { "urls": "stun:stun.sipwises.com:3478" },
        { "urls": "stun:stun.sipzone.net:3478" },
        { "urls": "stun:stun.sip.zap.com:3478" },
        { "urls": "stun:stun.sky.com:3478" },
        { "urls": "stun:stun.smartvoip.com:3478" },
        { "urls": "stun:stun.tango.com:3478" },
        { "urls": "stun:stun.talk.co.za:3478" },
        { "urls": "stun:stun.talktalk.net:3478" },
        { "urls": "stun:stun.tcp.com:3478" },
        { "urls": "stun:stun.tel.winsite.com:3478" },
        { "urls": "stun:stun.three.co.uk:3478" },
        { "urls": "stun:stun.tiscali.co.uk:3478" },
        { "urls": "stun:stun.trafficom.com:3478" },
        { "urls": "stun:stun.unifiedtelecom.com:3478" },
        { "urls": "stun:stun.unitymediagroup.com:3478" },
        { "urls": "stun:stun.usetel.com:3478" },
        { "urls": "stun:stun.vitalwerk.eu:3478" },
        { "urls": "stun:stun.voip.bh:3478" },
        { "urls": "stun:stun.voip.blue:3478" },
        { "urls": "stun:stun.voip.ch:3478" },
        { "urls": "stun:stun.voip.me:3478" },
        { "urls": "stun:stun.voip.mobi:3478" },
        { "urls": "stun:stun.voip.multicast.net:3478" },
        { "urls": "stun:stun.voip.purple.co.uk:3478" },
        { "urls": "stun:stun.voip.soundtalk.de:3478" },
        { "urls": "stun:stun.voipstunt.com:3478" },
        { "urls": "stun:stun.voiptel.net:3478" },
        { "urls": "stun:stun.voiptel.com:3478" },
        { "urls": "stun:stun.waupple.com:3478" },
        { "urls": "stun:stun.webuip.com:3478" },
        { "urls": "stun:stun.webwatcher.com:3478" },
        { "urls": "stun:stun.wesay.com:3478" },
        { "urls": "stun:stun.yourcall.com:3478" },
        { "urls": "stun:stun.zayo.com:3478" },
        { "urls": "stun:stun.znet.com:3478" },
        { "urls": "stun:stun.zconnect.net:3478" },
        { "urls": "stun:stun.zsmartvoip.com:3478" }
    ]
})


# Use WebRTC for live video stream
webrtc_streamer(key="example", video_processor_factory=lambda: CustomVideoProcessor(model))
