# Tạo thư mục làm việc cho convNet
!mkdir -p /kaggle/working/convNet

# Sao chép toàn bộ thư mục từ Dataset sang thư mục làm việc
!cp -r /kaggle/input/convnet/* /kaggle/working/convNet/

# Thay đổi thư mục hiện tại thành thư mục làm việc
%cd /kaggle/working/convNet/