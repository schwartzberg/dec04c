"""Notebook drawing interface; run after handwriting_helpers.py in a notebook."""
import anywidget
import traitlets
from IPython.display import display

class DigitDrawingBox(anywidget.AnyWidget):
    """A browser canvas that sends the user's drawing to Python on Save."""
    request = traitlets.Dict().tag(sync=True)
    status = traitlets.Unicode("Choose the true digit, draw it, then save. Suggested digits: 2, 5 and 8.").tag(sync=True)
    # JavaScript handles mouse/stylus strokes; Python handles inference and saving.
    _esm = """
    export default { render({model, el}) {
      const panel = document.createElement('div');
      panel.style.cssText = 'padding:16px;border:1px solid #aaa;border-radius:8px;background:#f6f7f8;color:#111;max-width:600px';
      const instructions = document.createElement('p');
      instructions.textContent = 'Draw one digit in black. Leave a margin. Save three different digits, keeping mistakes too.';
      const labelText = document.createElement('label');
      labelText.textContent = 'The digit I intend to draw: ';
      const select = document.createElement('select');
      select.setAttribute('aria-label', 'True digit');
      for (let i=0; i<10; i++) {
        const option = document.createElement('option'); option.value=String(i); option.textContent=String(i); select.append(option);
      }
      select.value='2'; labelText.append(select);
      const canvas = document.createElement('canvas');
      canvas.width=280; canvas.height=280;
      canvas.setAttribute('aria-label','Draw a handwritten digit here');
      canvas.style.cssText='display:block;width:280px;height:280px;border:2px solid #555;margin:12px 0;touch-action:none;cursor:crosshair;background:white';
      const ctx=canvas.getContext('2d');
      let drawing=false, hasInk=false;
      function clear() { ctx.fillStyle='white';ctx.fillRect(0,0,280,280);hasInk=false; }
      function position(event) { const r=canvas.getBoundingClientRect(); return [(event.clientX-r.left)*280/r.width,(event.clientY-r.top)*280/r.height]; }
      canvas.addEventListener('pointerdown', event => {
        event.preventDefault();canvas.setPointerCapture(event.pointerId);drawing=true;hasInk=true;
        const [x,y]=position(event);ctx.strokeStyle='black';ctx.lineWidth=18;ctx.lineCap='round';ctx.lineJoin='round';
        ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+0.01,y+0.01);ctx.stroke();
      });
      canvas.addEventListener('pointermove', event => { if(!drawing)return;const [x,y]=position(event);ctx.lineTo(x,y);ctx.stroke(); });
      canvas.addEventListener('pointerup', () => { drawing=false; });
      canvas.addEventListener('pointercancel', () => { drawing=false; });
      const clearButton=document.createElement('button');clearButton.textContent='Clear canvas';clearButton.addEventListener('click',clear);
      const saveButton=document.createElement('button');saveButton.textContent='Save and predict';saveButton.style.marginLeft='10px';
      const status=document.createElement('p');status.setAttribute('role','status');
      const refresh=()=> {status.textContent=model.get('status');saveButton.disabled=false;};
      saveButton.addEventListener('click', () => {
        if(!hasInk){status.textContent='Draw a digit first.';return;}
        saveButton.disabled=true;status.textContent='Saving and predicting...';
        // Send image and true label together, with a unique request id.
        model.set('request',{image:canvas.toDataURL('image/png'),label:Number(select.value),id:Date.now()+Math.random()});
        model.save_changes();
      });
      model.on('change:status',refresh);
      panel.append(instructions,labelText,canvas,clearButton,saveButton,status);el.append(panel);clear();refresh();
      return () => {model.off('change:status',refresh);};
    }};
    """


def save_digit_request(change):
    """Preserve each original drawing and record its inference result."""
    request = change["new"]
    if not request:
        return
    try:
        label = int(request["label"])
        if label not in range(10):
            raise ValueError("Choose a digit from 0 to 9.")
        data_url = request["image"]
        if not data_url.startswith("data:image/png;base64,"):
            raise ValueError("The drawing must be a PNG image.")
        drawing = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")
        prepared, predicted, score = predict_handwriting(drawing)
        # Number files rather than overwriting earlier attempts.
        number = 1
        while (HANDWRITING_DIR / f"digit_{label}_{number:02d}.png").exists():
            number += 1
        path = HANDWRITING_DIR / f"digit_{label}_{number:02d}.png"
        drawing.save(path)
        result = {"image": path.name, "true_digit": label, "predicted_digit": predicted,
                  "model_score": score, "correct": predicted == label,
                  "source": "user drawing in notebook", "checkpoint": "outputs/mnist/mnist_cnn.pth"}
        path.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        drawing_box.status = (f"Saved {path.name}. You wrote {label}; model predicted {predicted} "
                              f"(score {score:.1%}). Clear the canvas for the next digit.")
    except Exception as exc:
        drawing_box.status = f"Could not save: {exc}"


drawing_box = DigitDrawingBox()
drawing_box.observe(save_digit_request, names="request")
display(drawing_box)