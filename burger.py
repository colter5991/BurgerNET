#!/usr/bin/python3
import os
import json
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import argparse
from functools import partial
import tensorflowjs as tfjs

#"[START_TITLE]"
#"Title of restaurant"
#"[END_TITLE]"
#"[START_SECTION]"
#"asdf"
#"[START_SUBTEXT]"
#"asdf"
#"[START_ITEM]"
#"HURRR"
#"[START_PRICE]"
#"PRICE"
#"[START_DESCRIPTION]"
#"DESCRIPTION"
#"[END]"

MAX_MENU_LENGTH = 20000
BATCH_SIZE=64
SEQUENCE_LENGTH = 2048

START_TITLE_CHAR = '\u0990'
END_TITLE_CHAR = '\u0991'
START_SECTION_CHAR = '\u0992'
START_SUBTEXT_CHAR = '\u0993'
START_ITEM_CHAR = '\u0994'
START_PRICE_CHAR = '\u0995'
START_DESCRIPTION_CHAR = '\u0996'
END_CHAR = '\u0997'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
mixed_precision.set_policy(mixed_precision.Policy("mixed_float16"))


def for_each_file_in_dir(input_dir, func):
  for item in os.listdir(input_dir):
    path = "{}/{}".format(input_dir, item)
    if os.path.isfile(path):
      func(path)
    else:
      print("Recursing into \"{}\"".format(path))
      for_each_file_in_dir(path, func)

class Menu:
  def __init__(self, **kwargs):
    if "filename" in kwargs:
      self._init_from_file(kwargs["filename"])
    elif "string" in kwargs:
      self._init_from_string(kwargs["string"])
    elif "obj" in kwargs:
      self.__init_from_dict(kwargs["obj"])
    else:
      raise ValueError("Menu constructor must have one the following kwargs: (filename, string, obj)")

  def _init_from_file(self, filename):
    with open(filename, "r", encoding="utf-8") as f:
      self._init_from_dict(json.load(f))

  def _init_from_dict(self, obj):
    self._dict = obj
    self._string = self._dict_to_string(obj)

  def _init_from_string(self, string):
    self._string = string
    self._dict = self._string_to_dict(string)

  @property
  def string(self):
    return self._string

  @property
  def json(self):
    return self._dict

  def _dict_to_string(self, menu):
    string = START_TITLE_CHAR
    string += menu["name"]
    string += END_TITLE_CHAR
    for section in menu["sections"]:
      string += START_SECTION_CHAR
      string += section["title"]
      if section.get("subtitle", None) is not None:
        string += START_SUBTEXT_CHAR
        string += section["subtitle"]

      for item in section["items"]:
        string += START_ITEM_CHAR
        string += item["name"]
        string += START_PRICE_CHAR
        string += item["price"]
        if item.get("description", None) is not None:
          string += START_DESCRIPTION_CHAR
          string += item["description"]

    string += END_CHAR
    return string

  def _string_to_dict(self, string):
    # Create the dict to store the menu
    menu = {}

    # Parse the title start char, discard everything before
    start_split = string.split(START_TITLE_CHAR)
    if (len(start_split) > 1):
      string = start_split[1]

    # Parse the END_CHAR that delimits the end of the menu. Discard everything afterwards
    end_split = string.split(END_CHAR)
    if len(end_split) > 1:
      string = end_split[0]

    # Find the END_TITLE_CHAR.
    # - Everything before is the restaurant name
    # - Everything afterwards is the menu sections
    end_title_split = string.split(END_TITLE_CHAR)
    name = end_title_split[0]
    rest = end_title_split[1]
    menu["name"] = name

    # Parse each menu section that starts with the START_SECTION_CHAR
    menu["sections"] = []
    for section in rest.split(START_SECTION_CHAR)[1:]:
      section_json = {}

      # Split based on START_ITEM_CHAR
      # Everything before the first start item char referrs to the section title/subtitle
      items = section.split(START_ITEM_CHAR)
      section_json['title'] = items[0].split(START_SUBTEXT_CHAR)[0]
      if START_SUBTEXT_CHAR in items[0]:
        section_json['subtitle'] = items[0].split(START_SUBTEXT_CHAR)[-1]

      # Everything after the first START_ITEM_CHAR is a menu item
      section_json["items"] = []
      for item in items[1:]:
        item_json = {}
        # Split based on PRICE_CHAR, everything before is the item's title
        price_split = item.split(START_PRICE_CHAR)
        item_json["title"] = price_split[0]
        # If the price char is not present, then set the price to nothing
        if len(price_split) < 2:
          item_json["price"] =""
        # Otherwise, if this item hasa description, make sure it doesnt appear in the price
        else:
          item_json["price"] = price_split[1].split(START_DESCRIPTION_CHAR)[0]
        # Populate the item description if its present
        if START_DESCRIPTION_CHAR in item:
          item_json["description"] = item.split(START_DESCRIPTION_CHAR)[-1]
        section_json["items"].append(item_json)
      menu["sections"].append(section_json)

    return menu

  def print(self):
    data = self.json
    print("==========================")
    print("{}".format(data['name']))
    print("==========================")
    for section in data['sections']:
      print("------------------------------")
      print("{}".format(section["title"]))
      if "subtitle" in section:
        print("({})".format(section["subtitle"]))
      print("------------------------------")
      for item in section["items"]:
        if "$" in item["price"]:
          print(" - {} ({})".format(item["title"], item["price"]),end="")
        else:
          print(" - {}".format(item["title"]),end="")
        if "description" in item.keys():
          print(": {}".format(item["description"]),end="")
        print("")

class Vocab:
  def __init__(self, filename, input_dir=None):
    self.filename = filename
    self.input_dir = input_dir
    if not os.path.exists(filename) or not os.path.isfile(filename):
      print("Vocab not found, generating new vocabulary")
      if input_dir is None:
        raise ValueError("Cannot generate a vocabulary, no input_dir provided")
      self.create_vocab(input_dir, filename)
    vocab_dir = open(filename, 'r').readline().strip("\n")
    if input_dir is not None and vocab_dir != os.path.abspath(input_dir):
      print("Vocabulary was created from directory \"{}\" but is needed for \"{}\"".format(vocab_dir, os.path.abspath(input_dir)))
      print("Regenerating vocabulary...")
      self.create_vocab(input_dir, filename)
    self.num2char, self.char2num = self.get_vocab(filename)
    self.size = len(self.num2char)

  def str_to_set(self, string, string_set) :
    for c in string:
      string_set.add(c)

  def add_file_to_vocab(self, file):
    # Skip all non json files
    if file[-5:] != ".json":
      return
    menu = Menu(filename=file)
    menu_json = menu.json
    self.str_to_set(menu_json["name"], self.vocab)
    for section in menu_json["sections"]:
      self.str_to_set(section['title'], self.vocab)
      self.str_to_set(section.get('subtitle', ""), self.vocab)
      for item in section["items"]:
        self.str_to_set(item['name'], self.vocab)
        self.str_to_set(item['price'], self.vocab)
        self.str_to_set(item.get("description",""), self.vocab)

  def create_vocab(self, input_dir, filename):
    self.vocab = set()
    for_each_file_in_dir(input_dir, self.add_file_to_vocab)
    if '\n' in self.vocab:
      self.vocab.remove('\n')
    with open(filename, "w") as f:
      f.write(os.path.abspath(input_dir) + "\n")
      f.write("".join(sorted(list(self.vocab))))

  def get_vocab(self, filename):
    vocab = sorted([c for c in "".join(open(filename, "r", encoding="utf-8").readlines()[1:])] + 
      [START_TITLE_CHAR, END_TITLE_CHAR, START_SECTION_CHAR, START_SUBTEXT_CHAR, START_ITEM_CHAR, START_PRICE_CHAR, START_DESCRIPTION_CHAR, END_CHAR] + ['\r', '\n']
    )
    return vocab, {char: index for index, char in enumerate(vocab)}

  def string_to_array(self, string):
    return [self.char2num[c] for c in string]

  def array_to_string(self, array):
    return "".join([self.num2char[i] for i in array])

  def to_js_array(self, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    string = "const vocab = ["
    chars = []
    for char in self.num2char:
      val = ord(char)
      chars.append("\"\\u{}{}{}{}\"".format(
        format((val // (16**3)) % 16, 'x'),
        format((val // (16**2)) % 16, 'x'),
        format((val // (16**1)) % 16, 'x'),
        format((val // (16**0)) % 16, 'x'),
      ))
    string += ",".join(chars) + "]\n"
    with open(output_dir + "/vocab.js", "w") as f:
      f.write(string)



def build_model(vocab, batch_size=1):
  model = keras.Sequential([
    keras.layers.Embedding(vocab.size, 64, batch_input_shape=[batch_size,None], dtype="float32"),
    keras.layers.LSTM(512, return_sequences=True, stateful=True),
    keras.layers.Dense(vocab.size)
  ])

  for layer in model.layers:
    print(layer.dtype)

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

  return model


def load_training_data(input_dir, vocab):
  all_data = []

  # Create helper function to read menu file into array
  def add_menu_to_dataset(all_data, filename):
    if filename[-5:] != ".json":
      return
    all_data += vocab.string_to_array(Menu(filename=filename).string)
  # Call helper function on all menus in dataset
  for_each_file_in_dir(input_dir, partial(add_menu_to_dataset, all_data))

  # Build python array into dataset
  print("Convering python array into dataset...")
  dataset = tf.data.Dataset.from_tensor_slices(all_data)
  dataset = dataset.batch(SEQUENCE_LENGTH+1, drop_remainder=True)
  dataset = dataset.map(lambda x: (x[:-1], x[1:]) )
  dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
  return dataset

def generate_menu(model, restaurant_name, vocab):
  model.reset_states()
  input_data = tf.expand_dims(vocab.string_to_array(START_TITLE_CHAR + restaurant_name + END_TITLE_CHAR + START_SECTION_CHAR),0)
  print(input_data.shape)
  generated = []
  for i in range(0, MAX_MENU_LENGTH):
    predictions = tf.squeeze(model(input_data),0)
    next_char = tf.random.categorical(predictions, num_samples=1)[0][-1]
    generated.append(vocab.array_to_string([next_char.numpy()]))
    input_data = tf.expand_dims([next_char],0)
    if vocab.array_to_string([next_char.numpy()]) == END_CHAR:
      break

  print("".join([START_TITLE_CHAR, restaurant_name, END_TITLE_CHAR] + generated))
  return Menu(string="".join([START_TITLE_CHAR, restaurant_name, END_TITLE_CHAR] + generated))

def train(input_dir, model_dir, epochs, force=False):
  try:
    os.makedirs(model_dir, exist_ok=force)
  except FileExistsError as e:
      print("The directory \"{}\" already exists. To overwrite the existing model, use the --force flag".format(model_dir))
      return
  vocab = Vocab(model_dir +"/vocab.txt", input_dir)

  # Load all of the training data
  data = load_training_data(input_dir, vocab)

  # Build the model and save the un-trained model to a file
  model = build_model(vocab, BATCH_SIZE)
  inference_model = build_model(vocab,1)
  inference_model.set_weights(model.get_weights())
  model.save(model_dir + "/untrained.h5")
  inference_model.save(model_dir + "/inference_untrained.h5")

  # Go ahead an train the model, saving checkpoints along the way
  model.fit(data, epochs=epochs, callbacks=[
    tf.keras.callbacks.ModelCheckpoint(
      filepath=(model_dir+ "/checkpoint_{epoch}"),
      save_weights_only=True)
  ])

  # Finally, save the completely trained model
  inference_model.set_weights(model.get_weights())
  model.save(model_dir + "/trained.h5")
  inference_model.save(model_dir + "/inference_trained.h5")

def load_model(model_dir, checkpoint=None):
  if checkpoint is not None:
    model = keras.models.load_model(model_dir + "/inference_untrained.h5")
    model.load_weights(model_dir + "/checkpoint_"+str(checkpoint))
    return model
  else:
    return keras.models.load_model(model_dir + "/inference_trained.h5")


def generate(model_dir, name, checkpoint=None):
  model = load_model(model_dir, checkpoint)
  vocab = Vocab(model_dir+"/vocab.txt")
  generate_menu(model, name, vocab).print()

def to_json(model_dir, output_dir, checkpoint=None):
  vocab = Vocab(model_dir+"/vocab.txt")
  vocab.to_js_array(output_dir)
  model = load_model(model_dir, checkpoint)
  tfjs.converters.save_keras_model(model, output_dir)



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate restuarant menus")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--train", nargs=1, help="Training dataset directory. The directory is recursivly parsed for all *.json files")
  group.add_argument("--generate", nargs=1, help="Generate a restuarnt menu from the given restaurant name")
  group.add_argument("--to_json", nargs=1, help="Convert the model in a json format at the given location")

  parser.add_argument("--model_dir", nargs=1, default="model", help="The directory to store/read checkpoints and the completed model from")
  parser.add_argument("--epochs", nargs=1, type=int, default=15, help="How many epochs to generate when training")
  parser.add_argument("--force", action="store_true", help="Allow overwriting of an existing model directory")
  parser.add_argument("--checkpoint", nargs=1, type=int, help="Load the model from a specific training checkpoint instead of the completly trained model")

  args = parser.parse_args()
  print(args.__dict__)
  if args.train is not None:
    train(args.train[0], args.model_dir, args.epochs[0], args.force is not None)
  elif args.generate is not None:
    generate(args.model_dir, args.generate[0], None if args.checkpoint is None else args.checkpoint[0])
  elif args.to_json is not None:
    to_json(args.model_dir, args.to_json[0], None if args.checkpoint is None else args.checkpoint[0])


