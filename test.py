# 
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import cvzone
# import os
# 
# # Kiểm tra file tồn tại
# if not os.path.exists('best.pt'):
#     print("Lỗi: File best (2).pt không tồn tại!")
#     exit()
# if not os.path.exists('cityRoad_potHoles.mp4'):
#     print("Lỗi: File P.mp4 không tồn tại!")
#     exit()
# 
# # Tải mô hình YOLO
# model = YOLO('best.pt')
# class_names = model.names
# print("Loại mô hình:", model.task)  # In 'segment' hoặc 'detect'
# print("Danh sách lớp:", class_names)  # In danh sách tên lớp
# 
# # Mở file video
# cap = cv2.VideoCapture('cityRoad_potHoles.mp4')
# if not cap.isOpened():
#     print("Lỗi: Không thể mở P.mp4. Kiểm tra file hoặc codec.")
#     exit()
# 
# # Lấy FPS để đặt độ trễ
# fps = cap.get(cv2.CAP_PROP_FPS)
# delay = int(1000 / fps) if fps > 0 else 1
# print("FPS video:", fps, "Độ trễ:", delay, "ms")
# 
# frame_count = 0
# while True:
#     ret, img = cap.read()
#     if not ret:
#         print("Hết video hoặc lỗi đọc khung hình.")
#         break
#     
#     frame_count += 1
#     img = cv2.resize(img, (1020, 500))
#     h, w, _ = img.shape
# 
#     # Dự đoán với YOLO
#     results = model.predict(img)
#     print(f"Khung {frame_count}: Số kết quả: {len(results)}")
# 
#     for r in results:
#         boxes = r.boxes
#         masks = r.masks
#         print(f"  Số hộp: {len(boxes)}, Có mask: {masks is not None}")
# 
#         # Xử lý phân đoạn (nếu có mask)
#         if masks is not None:
#             masks = masks.data.cpu()
#             print(f"  Kích thước masks: {masks.shape}")
#             for seg, box in zip(masks.data.cpu().numpy(), boxes):
#                 seg = cv2.resize(seg, (w, h))
#                 contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 print(f"    Số contours: {len(contours)}")
#                 for contour in contours:
#                     d = int(box.classes)
#                     c = class_names[d] if d < len(class_names) else "Unknown"
#                     x, y, w, h = cv2.boundingRect(contour)
#                     cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
#                     label_y = max(10, y)
#                     cv2.putText(img, c, (x, label_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Nhãn xanh lá
#                     print(f"    Vẽ nhãn: {c} tại ({x}, {label_y - 10})")
# 
#         # Xử lý phát hiện (nếu không có mask)
#         if boxes and not masks:
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].int().tolist()
#                 d = int(box.cls)
#                 c = class_names[d] if d < len(class_names) else "Unknown"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(img, c, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                 print(f"    Vẽ hộp và nhãn: {c} tại ({x1}, {max(10, y1 - 10)})")
# 
#     cv2.imshow('Video', img)
#     if cv2.waitKey(delay) & 0xFF == ord('q'):
#         break
# 
# cap.release()
# cv2.destroyAllWindows()




from ultralytics import YOLO
import cv2
import numpy as np
import cvzone
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Kiểm tra file tồn tại
if not os.path.exists('best.pt'):
    print("Lỗi: File best.pt không tồn tại!")
    exit()
if not os.path.exists('p.mp4'):
    print("Lỗi: File cityRoad_potHoles.mp4 không tồn tại!")
    exit()

# Tải mô hình YOLO
model = YOLO('best.pt')
class_names = model.names
print("Loại mô hình:", model.task)  # In 'segment' hoặc 'detect'
print("Danh sách lớp:", class_names)  # In danh sách tên lớp

# Mở file video
cap = cv2.VideoCapture('p.mp4')
if not cap.isOpened():
    print("Lỗi: Không thể mở cityRoad_potHoles.mp4. Kiểm tra file hoặc codec.")
    exit()

# Lấy FPS để đặt độ trễ
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 1
print("FPS video:", fps, "Độ trễ:", delay, "ms")

# Danh sách để lưu nhãn thực tế và dự đoán
y_true = []  # Nhãn thực tế
y_pred = []  # Nhãn dự đoán

frame_count = 0
while True:
    ret, img = cap.read()
    if not ret:
        print("Hết video hoặc lỗi đọc khung hình.")
        break
    
    frame_count += 1
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape

    # Giả lập nhãn thực tế (vì có thể bạn chưa có tệp nhãn)
    # Giả sử 50% khung hình đầu tiên là Pothole (0), 50% còn lại là Smooth (1)
    true_label = 0 if frame_count <= 100 else 1  # Điều chỉnh tùy theo số khung hình video
    y_true.append(true_label)

    # Dự đoán với YOLO
    results = model.predict(img, conf=0.25)  # Giảm ngưỡng để tăng khả năng phát hiện
    print(f"Khung {frame_count}: Số kết quả: {len(results)}")

    pred_labels = []
    for r in results:
        boxes = r.boxes
        masks = r.masks
        print(f"  Số hộp: {len(boxes)}, Có mask: {masks is not None}")

        # Xử lý phân đoạn (nếu có mask)
        if masks is not None:
            masks = masks.data.cpu()
            print(f"  Kích thước masks: {masks.shape}")
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"    Số contours: {len(contours)}")
                for contour in contours:
                    d = int(box.cls)
                    pred_labels.append(d)  # Lưu nhãn dự đoán
                    c = class_names[d] if d < len(class_names) else "Unknown"
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    label_y = max(10, y)
                    cv2.putText(img, c, (x, label_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print(f"    Vẽ nhãn: {c} tại ({x}, {label_y - 10})")

        # Xử lý phát hiện (nếu không có mask)
        if boxes and masks is None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                d = int(box.cls)
                pred_labels.append(d)  # Lưu nhãn dự đoán
                c = class_names[d] if d < len(class_names) else "Unknown"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, c, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"    Vẽ hộp và nhãn: {c} tại ({x1}, {max(10, y1 - 10)})")

    # Nếu không có dự đoán nào, giả sử nhãn dự đoán là "Smooth" (1)
    pred_label = pred_labels[0] if pred_labels else 1
    y_pred.append(pred_label)

    print(f"Frame {frame_count}: True label: {true_label}, Predicted label: {pred_label}")

    cv2.imshow('Video', img)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Kiểm tra dữ liệu trước khi tạo ma trận
print(f"Total frames processed: {frame_count}")
print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")

# Tạo ma trận nhầm lẫn
if len(y_true) > 0 and len(y_pred) > 0:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 0: Pothole, 1: Smooth

    # Trực quan hóa ma trận nhầm lẫn
    labels = ['Pothole', 'Smooth']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': ''})

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix for YOLO Pothole Detection')
    plt.savefig('confusion_matrix.png')  # Lưu biểu đồ thành tệp
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()

    # In ma trận nhầm lẫn
    print("Confusion Matrix:")
    print(cm)
else:
    print("Không có dữ liệu để tạo ma trận nhầm lẫn. Vui lòng kiểm tra video và mô hình.")