from TTS.tts.utils.text.symbols import make_symbols, parse_symbols
from TTS.utils.generic_utils import find_module


def setup_model(config, speaker_manager: "SpeakerManager" = None, language_manager: "LanguageManager" = None, style_manager: "StyleManager" = None):
    print(" > Using model: {}".format(config.model))
    # fetch the right model implementation.
    if "base_model" in config and config["base_model"] is not None:
        MyModel = find_module("TTS.tts.models", config.base_model.lower())
    else:
        MyModel = find_module("TTS.tts.models", config.model.lower())
    # define set of characters used by the model
    if config.characters is not None:
        # set characters from config
        if hasattr(MyModel, "make_symbols"):
            symbols = MyModel.make_symbols(config)
        else:
            symbols, phonemes = make_symbols(**config.characters)
    else:
        from TTS.tts.utils.text.symbols import phonemes, symbols  # pylint: disable=import-outside-toplevel

        if config.use_phonemes:
            symbols = phonemes
        # use default characters and assign them to config
        config.characters = parse_symbols()
    # consider special `blank` character if `add_blank` is set True
    num_chars = len(symbols) + getattr(config, "add_blank", False)
    config.num_chars = num_chars
    # compatibility fix
    if "model_params" in config:
        config.model_params.num_chars = num_chars
    if "model_args" in config:
        config.model_args.num_chars = num_chars
    if config.model.lower() in ["vits"]:  # If model supports multiple languages
        model = MyModel(config, speaker_manager=speaker_manager, language_manager=language_manager)
    elif config.model.lower() in ["stylefast_pitch"] and config.style_encoder_config.use_supervised_style:
        model = MyModel(config, speaker_manager=speaker_manager, style_manager=style_manager)
    else:
        model = MyModel(config, speaker_manager=speaker_manager)
    return model


# TODO; class registery
# def import_models(models_dir, namespace):
# for file in os.listdir(models_dir):
# path = os.path.join(models_dir, file)
# if not file.startswith("_") and not file.startswith(".") and (file.endswith(".py") or os.path.isdir(path)):
# model_name = file[: file.find(".py")] if file.endswith(".py") else file
# importlib.import_module(namespace + "." + model_name)
#
#
## automatically import any Python files in the models/ directory
# models_dir = os.path.dirname(__file__)
# import_models(models_dir, "TTS.tts.models")
