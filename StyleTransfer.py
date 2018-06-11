from style_model.transfer_style import Stylizer
from style_model.optimizers.l_bfgs import L_BFGS
from style_model.util.build_callback import build_callback
from style_model.util.img_util import load_image
import glob


CONTENT = 'img/content/me.jpg'



size_content = load_image(CONTENT).size
dims = tuple([int(x/2) for x in size_content])

sty = Stylizer(content_weight=1, style_weight=1e5)

for path_style_img in glob.glob('img/styles/*.jpg'):
    print(path_style_img)
    name_style_img = path_style_img.split('/')[-1]
    seated_nudes = sty(
        content_path=CONTENT,
        style_path=path_style_img,
        optimize=L_BFGS(max_evaluations=20),
        iterations=10,
        image_size=dims,
        initialization_strat='content',
        callback=build_callback('build/me_transfer/' + name_style_img)
    )

    seated_nudes.save('img/me_transfer/' + name_style_img)
