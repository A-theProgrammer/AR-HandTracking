try:
    import pywavefront
except ModuleNotFoundError:
    print("pywavefront module not found. Please install it using: pip install PyWavefront")
    exit(1)


import moderngl
import moderngl_window as mglw
from pyrr import Matrix44


import cv2
import numpy as np
import os


from prediction import (
    predict,
    draw_landmarks_on_image,
    solvepnp,
    reproject,
    get_camera_matrix,
    get_fov_y,
)


class CameraAR(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "CameraAR"
    resource_dir = os.path.normpath(os.path.join(__file__, "../data"))


    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        self.prog3d = self.ctx.program(
            vertex_shader="""
                #version 330 core
                uniform mat4 Mvp;
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;
                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;
                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_vert = in_position;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                }
            """,
            fragment_shader="""
                #version 330 core
                uniform vec3 Color;
                uniform vec3 Light;
                uniform sampler2D Texture;
                uniform bool withTexture;
                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;
                out vec4 f_color;
                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    if (withTexture) {
                        f_color = vec4(Color * texture(Texture, v_text).rgb * lum, 1.0);
                    } else {
                        f_color = vec4(Color * lum, 1.0);
                    }
                }
            """,
        )
        self.mvp = self.prog3d["Mvp"]
        self.light = self.prog3d["Light"]
        self.color = self.prog3d["Color"]
        self.withTexture = self.prog3d["withTexture"]


        try:
            self.scene_cube = self.load_scene("crate.obj")
        except Exception as e:
            print(f"Error loading crate.obj: {e}")
            exit(1)
        try:
            self.scene_marker = self.load_scene("marker.obj")
        except Exception as e:
            print(f"Error loading marker.obj: {e}")
            exit(1)


        self.vao_cube = self.scene_cube.root_nodes[0].mesh.vao.instance(self.prog3d)
        self.vao_marker = self.scene_marker.root_nodes[0].mesh.vao.instance(self.prog3d)


        self.texture = self.load_texture_2d("crate.png")


        self.object_pos = np.array([0.0, 0.0, -30.0], dtype=np.float32)


        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise RuntimeError("Failed to open the camera.  Make sure a camera is connected.")


        ret, frame = self.capture.read()
        if not ret:
            raise RuntimeError("Failed to read from OpenCV capture device.")


        self.aspect_ratio = float(frame.shape[1]) / frame.shape[0]


        self.window_size = (int(720.0 * self.aspect_ratio), 720)


        self.video_texture = self.ctx.texture(
            self.window_size,
            3,
            np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.ubyte),
        )
        self.video_texture.build_mipmaps()
        self.video_texture.repeat_x = False
        self.video_texture.repeat_y = False
        self.video_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)


        self.video_prog = self.ctx.program(
            vertex_shader="""
                #version 330 core
                in vec2 in_vert;
                in vec2 in_text;
                out vec2 v_text;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_text = in_text;
                }
            """,
            fragment_shader="""
                #version 330 core
                uniform sampler2D Video;
                in vec2 v_text;
                out vec4 f_color;
                void main() {
                    f_color = texture(Video, v_text);
                }
            """,
        )


        vertices = np.array(
            [


                -1.0, -1.0,  0.0, 0.0,
                 1.0, -1.0,  1.0, 0.0,
                -1.0,  1.0,  0.0, 1.0,
                -1.0,  1.0,  0.0, 1.0,
                 1.0, -1.0,  1.0, 0.0,
                 1.0,  1.0,  1.0, 1.0,
            ],
            dtype="f4",
        )
        self.video_vbo = self.ctx.buffer(vertices.tobytes())
        self.video_vao = self.ctx.vertex_array(
            self.video_prog,
            [(self.video_vbo, "2f 2f", "in_vert", "in_text")],
        )


        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)


        self.pinched = False
        self.distance_to_object = 0.0


        self.smooth_factor = 0.7


    def render(self, time: float, frame_time: float):
        """
        Called every frame by ModernGL. We:
         1) Grab the camera frame with OpenCV
         2) Run detection using prediction.py
         3) Draw 2D landmarks + text (pinch info, dist info) into the frame
         4) Upload that frame as a texture in the background
         5) Use the 3D landmarks to update the cube's position if pinched
         6) Render the marker (optional) and the 3D cube
        """
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)


        ret, frame = self.capture.read()
        if not ret:
            return  


        frame = cv2.resize(frame, (self.window_size[0], self.window_size[1]))


        frame = cv2.flip(frame, 1)


        detection_result = predict(np.ascontiguousarray(frame[..., ::-1]))  


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = draw_landmarks_on_image(frame_rgb, detection_result)


        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


        self.pinched = False
        self.distance_to_object = 0.0


        if self.camera_matrix is None:
            self.camera_matrix = get_camera_matrix(frame_bgr, frame_bgr.shape[0])


        if (
            detection_result
            and detection_result.hand_world_landmarks
            and detection_result.hand_landmarks
        ):


            model_landmarks_list = detection_result.hand_world_landmarks[0]
            image_landmarks_list = detection_result.hand_landmarks[0]


            world_landmarks_list = solvepnp(
                [model_landmarks_list],
                [image_landmarks_list],
                self.camera_matrix,
                frame_bgr.shape[1],
                frame_bgr.shape[0],
            )


            reprojection_error, reprojection_points_list = reproject(
                world_landmarks_list,
                [image_landmarks_list],
                self.camera_matrix,
                frame_bgr.shape[1],
                frame_bgr.shape[0],
            )


            if reprojection_points_list:
                for l in reprojection_points_list[0]:
                    cv2.circle(frame_bgr, (int(l[0]), int(l[1])), 5, (0, 0, 255), 2)


            if len(model_landmarks_list) >= 9:
                thumb_tip_3d = np.array([
                    model_landmarks_list[4].x,
                    model_landmarks_list[4].y,
                    model_landmarks_list[4].z,
                ])
                index_tip_3d = np.array([
                    model_landmarks_list[8].x,
                    model_landmarks_list[8].y,
                    model_landmarks_list[8].z,
                ])


                pinch_dist = np.linalg.norm(thumb_tip_3d - index_tip_3d)


                if pinch_dist < 0.03:
                    self.pinched = True


            if world_landmarks_list:


                idx_x = image_landmarks_list[8].x * frame_bgr.shape[1]
                idx_y = image_landmarks_list[8].y * frame_bgr.shape[0]


                pos_3d = np.array([self.object_pos[0], self.object_pos[1], self.object_pos[2], 1.0])


                w_obj = self.camera_matrix @ np.array([self.object_pos[0], self.object_pos[1], self.object_pos[2]])


                xyz = np.array([self.object_pos[0], self.object_pos[1], self.object_pos[2]], dtype=np.float32)
                uvw = self.camera_matrix @ xyz
                if uvw[2] > 0.001:
                    x_2d = uvw[0] / uvw[2]
                    y_2d = uvw[1] / uvw[2]


                    dist_2d = np.sqrt((idx_x - x_2d)**2 + (idx_y - y_2d)**2)
                    self.distance_to_object = dist_2d
                else:
                    self.distance_to_object = -1  


                if self.pinched:


                    new_x = index_tip_3d[0] * 100.0
                    new_y = -index_tip_3d[1] * 100.0


                    alpha = self.smooth_factor
                    self.object_pos[0] = alpha * self.object_pos[0] + (1.0 - alpha) * new_x
                    self.object_pos[1] = alpha * self.object_pos[1] + (1.0 - alpha) * new_y


        cv2.putText(
            frame_bgr,
            f"Pinched: {self.pinched}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame_bgr,
            f"DistToObject: {self.distance_to_object:.2f}"
            if isinstance(self.distance_to_object, (float, int))
            else "No hand",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )


        frame_rgb_final = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb_final = np.flipud(frame_rgb_final)  
        self.video_texture.write(frame_rgb_final.tobytes())
        self.video_texture.use(location=0)
        self.video_prog["Video"].value = 0


        self.ctx.disable(moderngl.DEPTH_TEST)
        self.video_vao.render()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)


        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)


        translate = Matrix44.from_translation(self.object_pos)
        rotate = Matrix44.from_y_rotation(np.sin(time) * 0.5 + 0.2)
        scale = Matrix44.from_scale((3.0, 3.0, 3.0))
        mvp = proj * translate * rotate * scale


        if self.pinched:
            self.color.value = (1.0, 0.0, 0.0)
        else:
            self.color.value = (1.0, 1.0, 1.0)


        self.light.value = (10.0, 10.0, 10.0)
        self.mvp.write(mvp.astype("f4"))
        self.withTexture.value = True


        self.texture.use()
        self.vao_cube.render()


    def on_render(self, time: float, frame_time: float):
        self.render(time, frame_time)


    def close(self):
        super().close()
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CameraAR.run()