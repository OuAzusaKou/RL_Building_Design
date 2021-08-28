###### 定义空间关系评价 ######
##值越小越接近要求

class Score_relation:
    def __init__(self, room1, room2,state,grid_size):
        #self.df = df
        self.grid_size = grid_size
        self.step_length = 6
        self.state = state
        self.room1 = room1
        self.room2 = room2
        #self.area = area
        self.x_distance = abs(self.state[self.room1, 0] - self.state[self.room2, 0])
        self.y_distance = abs(self.state[self.room1, 1] - self.state[self.room2, 1])
        # print(self.x_distance)
        self.sub_score = 0
    def reset_room(self,room1, room2,state,grid_size):
        self.grid_size = grid_size
        self.step_length = 6
        self.state = state
        self.room1 = room1
        self.room2 = room2
        #self.area = area
        self.x_distance = abs(self.state[self.room1, 0] - self.state[self.room2, 0])
        self.y_distance = abs(self.state[self.room1, 1] - self.state[self.room2, 1])
        # print(self.x_distance)
        self.sub_score = 0

    def need_seperated(self):  # 需要相离1
        x_distance_need = abs(self.state[self.room1, 2] + self.state[self.room2, 2]) / 2  # 预期距离
        y_distance_need = abs(self.state[self.room1, 3] + self.state[self.room2, 3]) / 2

        x_score = abs(self.x_distance - x_distance_need)
        y_score = abs(self.y_distance - y_distance_need)

        if self.x_distance - x_distance_need <= 0 and self.y_distance - y_distance_need <= 0:
            score = min([x_score, y_score]) + self.step_length
        elif self.x_distance - x_distance_need <= 0 and self.y_distance - y_distance_need > 0:
            score = x_score + self.step_length
        elif self.x_distance - x_distance_need > 0 and self.y_distance - y_distance_need <= 0:
            score = y_score + self.step_length
        elif self.x_distance - x_distance_need > 0 and self.y_distance - y_distance_need > 0:
            score = 0
        else:
            score = 0

        return score

    def need_externally_tangent(self):  # 需要外切2
        x_distance_need = abs(self.state[self.room1,2] + self.state[self.room2,2]) / 2  # 房间几何中心x轴目标距离
        y_distance_need = abs(self.state[self.room1,3] + self.state[self.room2,3]) / 2  # 房间几何中心y轴目标距离

        x_score = abs(self.x_distance - x_distance_need)  # 将现有几何中心距离同目标距离之差的绝对值，作为score
        y_score = abs(self.y_distance - y_distance_need)

        if self.x_distance - x_distance_need >= 0 and self.y_distance - y_distance_need >= 0:
            score = x_score + y_score + self.step_length
        elif self.x_distance - x_distance_need >= 0 and self.y_distance - y_distance_need < 0:
            score = x_score
        elif self.x_distance - x_distance_need < 0 and self.y_distance - y_distance_need >= 0:
            score = y_score
        elif self.x_distance - x_distance_need < 0 and self.y_distance - y_distance_need < 0:
            score = min([x_score, y_score])

        return score

    def need_intersected(self):  # 需要相交3
        room1_width = self.state[self.room1, 2]
        room1_depth = self.state[self.room1, 3]
        room2_width = self.state[self.room2, 2]
        room2_depth = self.state[self.room2, 3]

        x_distance_need_max = abs(room1_width + room2_width) / 2  # 预期最大距离
        y_distance_need_max = abs(room1_depth + room2_depth) / 2

        x_distance_need_min = abs(room1_width - room2_width) / 2  # 预期最小距离
        y_distance_need_min = abs(room1_depth - room2_depth) / 2

        x_score = abs(self.x_distance - x_distance_need_max)
        y_score = abs(self.y_distance - y_distance_need_max)
        x_score_min = abs(self.x_distance - x_distance_need_min)
        y_score_min = abs(self.y_distance - y_distance_need_min)

        if self.x_distance - x_distance_need_max >= 0 and self.y_distance - y_distance_need_max >= 0:
            score = x_score + y_score + self.step_length
        elif self.x_distance - x_distance_need_max < 0 and self.y_distance - y_distance_need_max >= 0:
            score = y_score + self.step_length
        elif self.x_distance - x_distance_need_max >= 0 and self.y_distance - y_distance_need_max < 0:
            score = x_score + self.step_length
        elif self.x_distance - x_distance_need_max < 0 and self.y_distance - y_distance_need_max < 0:
            area_room1 = int(room2_width * room1_depth)
            area_room2 = int(room2_width * room2_depth)
            area_min = min([area_room1, area_room2])

            if self.area - area_min >= 0:  # 如果面积等于最小房间面积，则判断该状态为内切或包含状态
                score = min([x_score_min, y_score_min]) + self.step_length
            else:
                score = 0
        ###############wait for to do #################3
        return score

    def need_internally_tangent(self):  # 需要内切4
        room1_width = self.state[self.room1, 2]
        room1_depth = self.state[self.room1, 3]
        room2_width = self.state[self.room2, 2]
        room2_depth = self.state[self.room2, 3]

        x_distance_need = abs(room1_width - room2_width) / 2  # 预期最小距离
        y_distance_need = abs(room1_depth - room2_depth) / 2

        x_score = abs(self.x_distance - x_distance_need)
        y_score = abs(self.y_distance - y_distance_need)

        score_distance = x_score + y_score  # 距离

        if (room1_width >= room2_width) & (room1_depth >= room2_depth):
            score_size = 0
        elif (room1_width < room2_width) & (room1_depth < room2_depth):
            score_size = 0
        else:
            score_size = min([abs(room1_width - room2_width), abs(room1_depth - room2_depth)])  # 边长之差的最小值

        score = score_distance + score_size

        return score

    def need_contain(self):  # 需要包含5
        room1_width = self.state[self.room1, 2]
        room1_depth = self.state[self.room1, 3]
        room2_width = self.state[self.room2, 2]
        room2_depth = self.state[self.room2, 3]

        x_distance_need = abs(room1_width - room2_width) / 2  # 预期最小距离
        y_distance_need = abs(room1_depth - room2_depth) / 2

        x_score = abs(self.x_distance - x_distance_need)
        y_score = abs(self.y_distance - y_distance_need)

        if room1_width >= room2_width and room1_depth >= room2_depth:
            score_size = 0
        elif room1_width < room2_width and room1_depth < room2_depth:
            score_size = 0
        else:
            score_size = min([abs(room1_width - room2_width) / 4, abs(room1_depth - room2_depth) / 4])  # 边长之差的最小值

        if self.x_distance - x_distance_need >= 0 and self.y_distance - y_distance_need >= 0:
            score = x_score + y_score + self.step_length + score_size
        elif self.x_distance - x_distance_need >= 0 and self.y_distance - y_distance_need < 0:
            score = x_score + self.step_length + score_size
        elif self.x_distance - x_distance_need < 0 and self.y_distance - y_distance_need >= 0:
            score = x_score + self.step_length + score_size
        elif self.x_distance - x_distance_need < 0 and self.y_distance - y_distance_need < 0:
            score = score_size

        return score

    def need_inside_boundary(self, boundary, room):  # 需要相交 + 内切 + 包含6
        boundary_width = self.grid_size[0]
        boundary_depth = self.grid_size[1]
        room_width = self.state[room,2]
        room_depth = self.state[room,3]

        x_distance = abs(self.grid_size[0]/2 - self.state[room, 0])
        y_distance = abs(self.grid_size[1]/2 - self.state[room, 1])

        x_distance_need = abs(boundary_width - self.state[room, 2]) / 2  # 预期最小距离
        y_distance_need = abs(boundary_depth - self.state[room, 3]) / 2

        x_score = abs(x_distance - x_distance_need)
        y_score = abs(y_distance - y_distance_need)

        if room_width - boundary_width > 0:
            score_width = room_width - boundary_width
        else:
            score_width = 0
        if room_depth - boundary_depth > 0:
            score_depth = room_depth - boundary_depth
        else:
            score_depth = 0

        if x_distance - x_distance_need > 0 and y_distance - y_distance_need > 0:
            score_distance = x_score + y_score
        elif x_distance - x_distance_need > 0 and y_distance - y_distance_need <= 0:
            score_distance = x_score
        elif x_distance - x_distance_need <= 0 and y_distance - y_distance_need > 0:
            score_distance = y_score
        elif x_distance - x_distance_need <= 0 and y_distance - y_distance_need <= 0:
            score_distance = 0

        score = score_distance + score_width + score_depth
        #print(score)
        return score

    def union_set(self):
        x_distance_need = abs(self.state[self.room1, 2] + self.state[self.room2, 2]) / 2  # 预期距离
        y_distance_need = abs(self.state[self.room1, 3] + self.state[self.room2, 3]) / 2
        weight = x_distance_need - self.x_distance
        if weight < 0:
            weight = 0
        height = y_distance_need - self.y_distance
        if height < 0:
            height = 0

        area = weight * height

        return area