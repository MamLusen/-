import os
import cv2
import numpy as np
def compare_shapes(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    _, thresh1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)

    contours1, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours1 or not contours2:
        return 0.0

    similarity = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I2, 0)
    return 1 - similarity

def detect_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])

    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    total_pixels = image.shape[0] * image.shape[1]
    green_pixels = cv2.countNonZero(green_mask)
    white_pixels = cv2.countNonZero(white_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    green_percent = green_pixels / total_pixels
    white_percent = white_pixels / total_pixels
    yellow_percent = yellow_pixels / total_pixels

    if white_percent > 0.1:
        kernel = np.ones((5, 5), np.uint8)
        white_dilated = cv2.dilate(white_mask, kernel, iterations=1)
        white_contours, _ = cv2.findContours(white_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(white_contours) >= 3:
            return "powdery_mildew"

    if yellow_percent > 0.2:
        return "chlorosis"

    return "healthy"


def get_recommendations(disease):
    recommendations = {
        "powdery_mildew": [
            "1. Удалите и уничтожьте сильно пораженные части растения",
            "2. Обработайте фунгицидами на основе серы или бикарбоната калия",
            "3. Улучшите циркуляцию воздуха вокруг растений",
            "4. Избегайте полива сверху, поливайте под корень",
            "5. Применяйте профилактические обработки в начале сезона"
        ],
        "chlorosis": [
            "1. Проверьте pH почвы (должен быть 6.0-6.5 для большинства растений)",
            "2. Внесите хелатные удобрения с железом",
            "3. Улучшите дренаж почвы",
            "4. Используйте органические удобрения",
            "5. Избегайте переувлажнения почвы"
        ],
        "healthy": [
            "1. Продолжайте текущий уход за растением",
            "2. Регулярно осматривайте на признаки заболеваний",
            "3. Поддерживайте оптимальные условия выращивания",
            "4. Проводите профилактические обработки",
            "5. Соблюдайте севооборот (для сельскохозяйственных культур)"
        ]
    }
    return recommendations.get(disease, ["Рекомендации не найдены"])


def main():
    # Пути к папкам
    database_path = "C:\\Users\\user\\Desktop\\the_database"
    camera_path = "C:\\Users\\user\\Desktop\\photos_from_the_camera\\image_camera.jpg"

    # Загрузка изображения с камеры
    camera_image = cv2.imread(camera_path)
    if camera_image is None:
        print("Ошибка: Не удалось загрузить изображение с камеры")
        return

    camera_image = cv2.resize(camera_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    database_images = {}
    for filename in os.listdir(database_path):
        if filename.endswith(".jpg"):
            name = os.path.splitext(filename)[0]
            path = os.path.join(database_path, filename)
            img = cv2.imread(path)
            if img is not None:
                database_images[name] = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    required_images = ['chlorosis', 'dandelion', 'healthy_leaf', 'plantain', 'powdery_mildew']
    for req in required_images:
        if req not in database_images:
            print(f"Ошибка: Отсутствует изображение {req} в базе данных")
            return

    weed_similarities = {
        "dandelion": compare_shapes(camera_image, database_images["dandelion"]),
        "plantain": compare_shapes(camera_image, database_images["plantain"])
    }

    max_weed = max(weed_similarities.items(), key=lambda x: x[1])

    if max_weed[1] > 0.7:
        print(f"Обнаружен сорняк: {max_weed[0]}")
        print("Рекомендации: удалите растение вручную или используйте гербициды")
        return

    color_result = detect_color(camera_image)
    recommendations = get_recommendations(color_result)

    if color_result == "healthy":
        print("Обнаружено здоровое растение")
    elif color_result == "powdery_mildew":
        print("Растение с заболеванием мучнистая роса. Рекомендации:")
    elif color_result == "chlorosis":
        print("Растение с заболеванием хлороз. Рекомендации:")

    for recommendation in recommendations:
        print(f"- {recommendation}")


if __name__ == "__main__":
    main()