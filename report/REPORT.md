# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Ultralytics

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không 

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): class_id=468, class_name=cab, rank=1, score=0.510915, taxonomy_name=ImageNet-1K.
- Record này mô tả toàn ảnh như thế nào? 
-> Đây là prediction cấp toàn ảnh (image-level). Model gán một nhãn duy nhất (hoặc top-k) cho cả bức ảnh, không chỉ định vị trí từng đối tượng. Với ảnh traffic, model chọn “cab” là lớp có xác suất cao nhất.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
-> Taxonomy/dataset huấn luyện của checkpoint. Ở đây là ImageNet-1K (1000 lớp). Checkpoint yolo11n-cls.pt chỉ có thể dự đoán trong danh sách lớp của taxonomy đó.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
-> class_id: định danh ổn định, máy đọc được, tránh lỗi chuỗi.
class_name: dễ đọc cho người.
taxonomy_name: xác định ngữ cảnh/lớp hợp lệ (ImageNet-1K ≠ COCO-80). Cần cả ba để tránh nhầm lẫn khi map, audit hoặc merge dữ liệu từ nhiều nguồn.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
-> Quy tắc chọn nhãn chính (dominant object / scene label), thứ tự ưu tiên khi nhiều lớp cạnh tranh, và khi nào dùng multi-label hoặc “other/unknown”. Cần ví dụ rõ ràng và tiêu chí quyết định (diện tích, vị trí trung tâm, ngữ cảnh…).
- Vì sao model score không phải ground truth? 
-> Score là độ tin cậy nội bộ của model (softmax probability), không phải nhãn đúng. Ground truth do annotator/reviewer gán theo guideline. Score có thể cao nhưng vẫn sai (ví dụ ảnh traffic bị gán “cab” trong khi cảnh là đường phố đông xe).

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    {
    "class_name": "person",
    "score": 0.912625,
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
    }
- Diễn giải vị trí box bằng lời: Box bao quanh đối tượng theo tọa độ pixel góc trên-trái (x1,y1) và góc dưới-phải (x2,y2). Ví dụ person đứng bên phải ảnh được bao bởi một hình chữ nhật lớn bao phủ toàn bộ thân và đầu.
- So sánh số prediction ở hai threshold: Threshold cao (ví dụ 0.5–0.6) → ít box hơn, chỉ giữ các dự đoán chắc chắn.
Threshold thấp (0.35 như trong file) → nhiều box hơn, bao gồm cả dự đoán yếu.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
->Threshold thấp tăng recall (độ bao phủ) nhưng tăng false positive → reviewer phải kiểm tra nhiều box hơn, tốn thời gian. Threshold cao giảm khối lượng review nhưng có thể bỏ sót object.
- Đề xuất một quy tắc box chặt: Box phải bao sát object (tight fit), không để thừa quá nhiều background; cạnh box không cắt vào phần chính của object; với object nhỏ/xa, vẫn phải bao đủ phần nhìn thấy.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
-> Đối với object bị che khuất/cắt mép, guideline cần quy định liệu có lấy cả phần bị che khuất vào box hay không nếu không quy định rõ hoặc tùy theo phần trăm bị che khuất thì cần escalation để quyết định có lấy cả phần bị che khuất đó hay không.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    {
    "instance_id": "kitchen-001",
    "class_name": "person",
    "score": 0.899318,
    "polygon_point_count": 348,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ],...
    }
- Polygon bổ sung chi tiết gì so với box?
-> Polygon mô tả biên dạng thực của object (shape), không chỉ hình chữ nhật bao quanh. Cho phép tính diện tích chính xác, tách instance chồng lên nhau, và hỗ trợ các task cần mask (matting, editing…).
- `instance_id` dùng để làm gì và không phải loại ID nào?
-> Dùng để định danh từng instance riêng biệt trong cùng một ảnh (phân biệt person A vs person B). Không phải class_id, không phải coco_image_id, và không phải ID toàn cục xuyên ảnh.
- Đề xuất một quy tắc biên mask: Mask phải bám sát biên thực của object; không bao gồm background; với vùng mờ/tiếp xúc phải quyết định rõ “belong to object A hoặc B”; pixel thuộc object thì gán, pixel background thì không.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
-> Guideline cần quy định cách xử lý soft edge, overlapping, occlusion (chỉ segment phần nhìn thấy hay ước lượng). Khi biên không rõ hoặc nhiều instance dính nhau → escalate để lead quyết định theo rule thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                | Đơn vị/định dạng ground truth          | Lỗi hoặc điểm mơ hồ quan sát được                  | Annotator làm gì?                          | Reviewer xem gì?                              |
| --------------------- | -------------------------------------- | -------------------------------------------------- | ------------------------------------------ | --------------------------------------------- |
| Phân loại ảnh         | 1 (hoặc multi) class_id/name theo taxonomy | Chọn nhãn sai khi nhiều chủ thể; score cao nhưng sai | Chọn nhãn theo guideline                   | Nhãn cuối + consistency với taxonomy          |
| Phát hiện vật thể     | List {class, bbox_xyxy}                | Box lỏng/chặt, bỏ sót, nhầm lớp, threshold          | Vẽ box + gán lớp                           | Box quality, missing/false positive, occlusion |
| Instance segmentation | List {class, polygon/mask, instance_id}| Biên mờ, overlapping, mask thừa/thiếu               | Vẽ polygon/mask + gán lớp + instance_id    | Chất lượng biên, tách instance, occlusion     |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không đưa dữ liệu cá nhân (PII), ảnh nhạy cảm hoặc dữ liệu ngoài phạm vi vào report/output; chỉ sử dụng dữ liệu được phép trong lab.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Người phụ trách lab / mentor / lead của khóa.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
