import cv2
from statistics import median


class ContourRect:
    def __init__(self, x, y, w, h) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContourRect):
            return False
        return self.x == other.x and self.y == other.y
    # 121, 12, 22, 30
    # 91, 12, 38, 33
    def intersects(self, other, threshold=0.15) -> bool:
        widths_overlap = self.x <= other.x+other.w and other.x <= self.x+self.w
        heights_overlap = self.y <= other.y+other.h and other.y <= self.y+self.h
        return widths_overlap and heights_overlap and self.intersection_percentage(self, other) > threshold


    @staticmethod
    def intersection_percentage(rect1, rect2):
        area1 = rect1.w * rect1.h
        area2 = rect2.w * rect2.h

        x1max = max(rect1.x, rect2.x)
        x2min = min(rect1.x+rect1.w, rect2.x+rect2.w)

        width_intersection = x2min-x1max
        
        y1max = max(rect1.y, rect2.y)
        y2min = min(rect1.y+rect1.h, rect2.y+rect2.h)

        height_intersection = y2min-y1max

        intersection_area = width_intersection * height_intersection

        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area




    def overlap(self, other):
        if not self.intersects(other, threshold=0.):
            raise ValueError(f'The two bounding rectangles don\'t intersect: {self.x, self.y, self.w, self.h} and {other.x, other.y, other.w, other.h}')
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x+self.w, other.x+other.w)
        y2 = max(self.y+self.h, other.y+other.h)

        self.x = x1
        self.y = y1
        self.w = x2-x1
        self.h = y2-y1
        return self

def select_roi(image, contours, intersect=True):
    clean_image = image.copy()
    height, width = image.shape[0], image.shape[1]

    sorted_regions = []
    regions_array = []
    contour_rects =[]


    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_rect = ContourRect(x, y, w, h)

        if is_contour_letter(image, contour):
            if intersect:
                intersecting_rects = list(filter(lambda rect: bounding_rect.intersects(rect), contour_rects))
                if len(intersecting_rects) > 0:
                    for rect in intersecting_rects:
                        try: bounding_rect.overlap(rect) 
                        except: print('kako')
                        contour_rects.remove(rect)
                    x, y, w, h = bounding_rect.x, bounding_rect.y, bounding_rect.w, bounding_rect.h
            contour_rects.append(bounding_rect)

    median_rect_size = contour_median_size(contour_rects) if len(contour_rects) != 0 else 0
    filtered_contour_rects = filter(
        lambda rect: is_rect_within_letter_height_range(rect, median_rect_size), 
        contour_rects
        )

    for rect in filtered_contour_rects:
        x, y, w, h = rect.x, rect.y, rect.w, rect.h
        region = clean_image[y:y+h+1, x:x+w+1]
        regions_array.append([region, (x, y, w, h)])
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda region_data: region_data[1][0])
    sorted_regions = [region[0] for region in regions_array]
    return image, sorted_regions

def is_contour_letter(image, contour):
    height, width = image.shape[0], image.shape[1]
    x, y, w, h = cv2.boundingRect(contour)

    is_contour_tall_enough = 0.3*height < h < 0.9*height
    is_contour_wide_enough = w < 0.5*width
    is_contour_letter_shaped = h > w and h < 4*w

    return (
        is_contour_tall_enough and
        is_contour_wide_enough and
        is_contour_letter_shaped
    )


def contour_median_size(contour_rects):
    median_rect_size = median([rect.h for rect in contour_rects])
    return median_rect_size

def is_rect_within_letter_height_range(contour_rect, median, threshold=0.12):
    allowed_percentage = 1-threshold  # in default case this is 0.88.

    lower_diff = median/contour_rect.h  # if contour_rect is larger than median 'letter' height
    higher_diff = contour_rect.h/median # if contour_rect is smaller than median 'letter' height

    diff = lower_diff if lower_diff < 1. else higher_diff
    return diff > allowed_percentage


def test():
    r1 = ContourRect(121, 12, 22, 30)
    r2 = ContourRect(91, 12, 38, 33)
    intersect = r1.intersects(r2)

if __name__ == '__main__':
    test()