#!/bin/bash
# Open-LLM-VTuber Kiosk Mode
# HDMI 모니터에만 전체화면으로 브라우저를 띄움

URL="${1:-http://localhost:12393}"

# PulseAudio 설정 (Discord 오디오 캡처용)
# user 1000의 PulseAudio에 TCP로 연결
export PULSE_SERVER=tcp:127.0.0.1

# X 서버 시작 + HDMI-1에만 출력
xinit /bin/bash -c "
    # HDMI-1만 켜고 메인 화면(eDP-1)은 끄기 (네이티브 해상도 사용)
    xrandr --output eDP-1 --off --output HDMI-1 --auto --primary
    sleep 1

    # HDMI 해상도 자동 감지
    RES=\$(xrandr | grep 'HDMI-1' | grep -oP '\d+x\d+\+')
    WIDTH=\$(echo \$RES | grep -oP '^\d+')
    HEIGHT=\$(echo \$RES | grep -oP 'x\K\d+')

    # 한글 입력기 (fcitx5) 시작
    export GTK_IM_MODULE=fcitx
    export QT_IM_MODULE=fcitx
    export XMODIFIERS=@im=fcitx
    fcitx5 -d &
    sleep 1

    # 마우스 커서 비활동 시 숨기기 (5초 후)
    unclutter -idle 5 &

    # VNC 서버 시작 (포트 5900, 메인 PC에서 원격 조작 가능)
    x11vnc -display :0 -forever -nopw -shared -rfbport 5900 \
        -threads -noxdamage -noxfixes \
        -defer 10 -wait 10 -snapfb \
        -ncache 10 &

    # Google Chrome 키오스크 모드로 실행
    PULSE_SERVER=tcp:127.0.0.1 google-chrome \
        --no-sandbox \
        --kiosk \
        --no-first-run \
        --disable-infobars \
        --disable-session-crashed-bubble \
        --disable-translate \
        --noerrdialogs \
        --incognito \
        --autoplay-policy=no-user-gesture-required \
        --window-size=\${WIDTH},\${HEIGHT} \
        --window-position=0,0 \
        --force-device-scale-factor=1 \
        '$URL'
" -- :0 vt1
