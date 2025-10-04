

document.addEventListener('DOMContentLoaded', function () {
  const trainForm = document.getElementById('train-form')
  const trainResult = document.getElementById('train-result')

  trainForm.addEventListener('submit', async function (e) {
    e.preventDefault()
    const formData = new FormData(trainForm)
    const body = new URLSearchParams()
    for (const pair of formData) { body.append(pair[0], pair[1]) }

    const res = await fetch('/train', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: body })
    const data = await res.json()
    trainResult.textContent = JSON.stringify(data, null, 2)
  })

  const jsonPredictBtn = document.getElementById('json-predict')
  const predictResult = document.getElementById('predict-result')
  const jsonInput = document.getElementById('json_input')

  jsonPredictBtn.addEventListener('click', async function (e) {
    e.preventDefault()
    let jsonText = jsonInput.value.trim()
    if (!jsonText) { predictResult.textContent = 'Please paste JSON representing a single row (object).'; return }
    try {
      const payload = JSON.parse(jsonText)
      const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      const data = await res.json()
      predictResult.textContent = JSON.stringify(data, null, 2)
    } catch (err) {
      predictResult.textContent = 'Invalid JSON: ' + err.message
    }
  })
})
