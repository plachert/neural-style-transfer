import torch
import tqdm



def gram_matrix(feature_maps):
    a, b, c, d = feature_maps.size()  
    features = feature_maps.view(a * b, c * d) 
    G = torch.mm(features, features.t()) 
    return G.div(a * b * c * d)

def maps_to_gram(list_of_feats):
    grams = [gram_matrix(feature_maps) for feature_maps in list_of_feats]
    return grams

def get_loss(ref_content, ref_style, image_content, image_style, content_weight=1, style_weight=100000):
    l_content = torch.nn.functional.mse_loss(ref_content, image_content)
    l_style = [torch.nn.functional.mse_loss(ref_style[i], image_style[i]) for i in range(len(ref_style))]
    l_style = torch.mean(torch.stack(l_style))
    print(l_content, l_style)
    loss = content_weight * l_content + style_weight * l_style
    return loss


def optimize_image(content_image, style_image, input_image, model, lr=0.1, n_iter=600, tv_coeff=10):
    image = np.copy(input_image.detach().numpy().squeeze())
    input_image = image_utils.prepare_input_image(image, requires_grad=True)
    model(content_image)
    ref_content = model.activations_values["content"][0]
    model(style_image)
    ref_style = maps_to_gram(model.activations_values["style"])
    optimizer = torch.optim.Adam([input_image], lr=lr)
    processed_images = []
    size = input_image.shape[-2] * input_image.shape[-1]
    
    for _ in tqdm(range(n_iter)):
        optimizer.zero_grad()
        model(input_image)
        image_content = model.activations_values["content"][0]
        image_style = maps_to_gram(model.activations_values["style"])
        tv = tv_coeff * total_variation(input_image) / size
        loss = get_loss(ref_content, ref_style, image_content, image_style) + tv
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            input_image.clamp_(0, 1)
        processed_images.append(np.copy(input_image.detach().numpy().squeeze()))
    return processed_images

