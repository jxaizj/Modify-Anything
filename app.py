import gradio as gr
from demo import lama_image_app, lama_video_app, change_video, change_image, change_aimage, change_avideo

available_lamamodels = ["lama", "ldm", "zits", "mat", "fcf", "manga", "sd2"]
default_lamammodel = "lama"


def image_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image_file = gr.Image(type="filepath").style(height=260)
                with gr.Row():
                    with gr.Column():
                        image_model_type = gr.Dropdown(
                            choices=[
                                "vit_h",
                                "vit_l",
                                "vit_b",
                            ],
                            value="vit_l",
                            label="Model Type",
                        )

                with gr.Row():
                    with gr.Column():
                        sahi_model_type = gr.Dropdown(
                            choices=[
                                "yolov5",
                                "yolov8",
                            ],
                            value="yolov8",
                            label="Detector Model Type",
                        )
                        sahi_image_size = gr.Slider(
                            minimum=0,
                            maximum=1600,
                            step=32,
                            value=640,
                            label="Image Size",
                        )

                        sahi_overlap_width = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.2,
                            label="Overlap Width",
                        )

                        sahi_slice_width = gr.Slider(
                            minimum=0,
                            maximum=640,
                            step=32,
                            value=256,
                            label="Slice Width",
                        )

                    with gr.Row():
                        with gr.Column():
                            sahi_model_path = gr.Dropdown(
                                choices=["yolov5l.pt", "yolov5l6.pt", "yolov8l.pt", "yolov8x.pt"],
                                value="yolov8l.pt",
                                label="Detector Model Path",
                            )
                            selected_lamamodel = gr.Dropdown(choices=available_lamamodels, label="lama Model(lama, ldm, zits, mat, fcf, mang, sd2)",
                                                             value=default_lamammodel,
                                                             interactive=True)

                            sahi_conf_th = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.2,
                                label="Confidence Threshold",
                            )
                            sahi_overlap_height = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.2,
                                label="Overlap Height",
                            )
                            sahi_slice_height = gr.Slider(
                                minimum=0,
                                maximum=640,
                                step=32,
                                value=256,
                                label="Slice Height",
                            )
                image_predict = gr.Button(value="Generate vector images from targets(去目标生成矢量图片)")

                with gr.Column():
                    output_image = gr.Gallery()

        image_predict.click(
            fn=lama_image_app,
            inputs=[
                image_file,
                image_model_type,
                selected_lamamodel,
                sahi_model_type,
                sahi_model_path,
                sahi_conf_th,
                sahi_image_size,
                sahi_slice_height,
                sahi_slice_width,
                sahi_overlap_height,
                sahi_overlap_width,
            ],
            outputs=[output_image],
        )
    with gr.Row():
        with gr.Column():
            b_image = gr.Image(type="filepath")
        with gr.Column():
            output_change = gr.Image()
    with gr.Row():
        change_images = gr.Button(value="Target and background image(目标与背景图)")
        change_images.click(
            fn=change_aimage,
            inputs=[image_file, b_image],
            outputs=[output_change]

        )
    with gr.Row():
        with gr.Column():
            b_video = gr.Video(type="filepath").style(height=260)

        with gr.Column():
            output_change = gr.Video()
    with gr.Row():
        change_videos = gr.Button(value="Target and Background Video(目标与背景视频)")
        change_videos.click(
            fn=change_avideo,
            inputs=[image_file, b_video],
            outputs=[output_change]

        )


def video_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                sahi_image_file = gr.Video().style(height=260)
                sahi_autoseg_model_type = gr.Dropdown(
                    choices=[
                        "vit_h",
                        "vit_l",
                        "vit_b",
                    ],
                    value="vit_l",
                    label="Sam Model Type",
                )

                with gr.Row():
                    with gr.Column():
                        sahi_model_type = gr.Dropdown(
                            choices=[
                                "yolov5",
                                "yolov8",
                            ],
                            value="yolov8",
                            label="Detector Model Type",
                        )
                        sahi_image_size = gr.Slider(
                            minimum=0,
                            maximum=1600,
                            step=32,
                            value=640,
                            label="Image Size",
                        )

                        sahi_overlap_width = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.2,
                            label="Overlap Width",
                        )

                        sahi_slice_width = gr.Slider(
                            minimum=0,
                            maximum=640,
                            step=32,
                            value=256,
                            label="Slice Width",
                        )

                    with gr.Row():
                        with gr.Column():
                            sahi_model_path = gr.Dropdown(
                                choices=["yolov5l.pt", "yolov5l6.pt", "yolov8l.pt", "yolov8x.pt"],
                                value="yolov8l.pt",
                                label="Detector Model Path",
                            )
                            selected_lamamodel = gr.Dropdown(choices=available_lamamodels, label="lama Model(lama, ldm, zits, mat, fcf, mang, sd2)",
                                                             value=default_lamammodel,
                                                             interactive=True)

                            sahi_conf_th = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.2,
                                label="Confidence Threshold",
                            )
                            sahi_overlap_height = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.2,
                                label="Overlap Height",
                            )
                            sahi_slice_height = gr.Slider(
                                minimum=0,
                                maximum=640,
                                step=32,
                                value=256,
                                label="Slice Height",
                            )
                sahi_image_predict = gr.Button(value="Generate vector video by removing targets(去目标生成矢量视频)")

            with gr.Column():
                output_video = gr.Video()
                output_video1 = gr.Video()

        sahi_image_predict.click(
            fn=lama_video_app,
            inputs=[
                sahi_image_file,
                sahi_autoseg_model_type,
                selected_lamamodel,
                sahi_model_type,
                sahi_model_path,
                sahi_conf_th,
                sahi_image_size,
                sahi_slice_height,
                sahi_slice_width,
                sahi_overlap_height,
                sahi_overlap_width,
            ],
            outputs=[output_video, output_video1],
        )
    with gr.Row():
        with gr.Column():
            b_image = gr.Image(type="filepath").style(height=260)

        with gr.Column():
            output_change = gr.Video()

    with gr.Row():
        change_images = gr.Button(value="Target and background image(目标与背景图)")
        change_images.click(
            fn=change_image,
            inputs=[sahi_image_file, b_image],
            outputs=[output_change])

    with gr.Row():
        with gr.Column():
            b_video = gr.Video()
        with gr.Column():
            output_change = gr.Video()
    with gr.Row():
        change_videos = gr.Button(value="Target and Background Video(目标与背景视频)")
        change_videos.click(
            fn=change_video,
            inputs=[sahi_image_file, b_video],
            outputs=[output_change]

        )


def metaseg_app():
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Image"):
                    image_app()

                with gr.Tab("Video"):
                    video_app()

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True)


if __name__ == "__main__":
    metaseg_app()
